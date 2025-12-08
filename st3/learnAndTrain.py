import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class MyWrapper(gym.Wrapper):
    def __init__(self):
        base_env = gym.make('CartPole-v1')
        monitored_env = Monitor(base_env)
        super().__init__(monitored_env)
        self.env = monitored_env

    def reset(self, *, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        return state, info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info
    
env=MyWrapper()

model = PPO('MlpPolicy', env, verbose=0)

print("Starting evaluation...")

# 优先学习再评估,学习的步数可以根据需要调整，越大越好
model.learn(total_timesteps=40000,progress_bar=True)

# 第一个参数是训练好的模型，第二个参数是环境，第三个参数是评估的回合数
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)

# 评估结果是平均奖励和标准差
print(f"Evaluation finished: mean reward={mean_reward:.2f}, std={std_reward:.2f}")
