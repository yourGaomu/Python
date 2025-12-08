import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy 


class MyWrapper(gym.Wrapper):
    def __init__(self):
        base_env = gym.make('CartPole-v1')
        super().__init__(base_env)
        self.env = base_env

    def reset(self, *, seed=None, options=None):
        # 重写reset方法
        # 第一个参数是随机种子，第二个参数是环境选项
        state, info = self.env.reset(seed=seed, options=options)
        # 返回状态和信息
        return state, info
    
    def step(self, action):
        # 重写step方法
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info

def text_env(env):
    print('env.observation_space:', env.observation_space)
    print('env.action_space:', env.action_space)

    state, info = env.reset()
    print('reset state:', state)
    print('reset info:', info)
    action = env.action_space.sample()
    print('sample action:', action)
    next_state, reward, terminated, truncated, info = env.step(action)
    print('step next_state:', next_state)


if __name__ == "__main__":
    # 创建环境包装器实例
    sample_env = MyWrapper()
    text_env(sample_env)
    sample_env.close()
    # 4个环境一起训练
    env = make_vec_env(MyWrapper, n_envs=4)
    # 重置环境
    env.reset()
    if torch.cuda.is_available():
        print("当前环境支持GPU")
        model = PPO('MlpPolicy', env
                    ,n_steps=1024
                    ,batch_size=64
                    ,n_epochs=4
                    ,gamma=0.99
                    ,gae_lambda=0.98
                    ,ent_coef=0.01
                    ,verbose=1
                    , device="cuda")
    else:
        print("当前环境不支持GPU，使用CPU")
        model = PPO('MlpPolicy'
                    , env
                    ,n_steps=1024
                    ,batch_size=64
                    ,n_epochs=4
                    ,gamma=0.99
                    ,gae_lambda=0.98
                    ,ent_coef=0.01
                    ,verbose=1
                    , device="cpu")

    # 先测试没有学习的模型
    mean_reward, std_reward =  evaluate_policy(model, env, n_eval_episodes=10)
    print(f"当前的期望: {mean_reward} 当前的标准差: {std_reward}")
    print("开始学习了...")
    # 优先学习再评估,学习的步数可以根据需要调整
    model.learn(total_timesteps=30000,progress_bar=True)
    # 第一个参数是训练好的模型，第二个参数是环境，第三个参数是评估的回合数
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"学习后的期望: {mean_reward} 学习后的标准差: {std_reward}")
