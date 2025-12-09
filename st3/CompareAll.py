import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")
import stable_baselines3 as sb3
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO,A2C,DQN,SAC,TD3,DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
import ale_py
from concurrent.futures import ProcessPoolExecutor
from time import sleep
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os
import numpy as np
import math

# 算法	支持的动作空间	你的环境	结果
# PPO	离散 + 连续	    离散 ✓	能运行
# A2C	离散 + 连续	    离散 ✓	能运行
# DQN	只支持离散	    离散 ✓	能运行
# SAC	只支持连续	    离散 ✗	不能运行
# TD3	只支持连续	    离散 ✗	不能运行
# DDPG	只支持连续	    离散 ✗	不能运行


class MountainCarRewardWrapper(gym.Wrapper):
    """
    自定义奖励包装器：根据位置和速度给予更密集的奖励，加速收敛
    公式参考：
    if x >= 0.5: reward = 1000
    if -0.5 < x < 0.5: reward = 2^(5*(x+1)) + (100*|v|)^2
    if x <= -0.5: reward = 100*|v|
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # obs[0]: 位置, obs[1]: 速度
        position = obs[0]
        velocity = obs[1]
        
        if position >= 0.5:
            new_reward = 1000.0
        elif -0.5 < position < 0.5:
            new_reward = math.pow(2, 5 * (position + 1)) + (100 * abs(velocity)) ** 2
        else: # position <= -0.5
            new_reward = 100 * abs(velocity)
            
        return obs, new_reward, terminated, truncated, info

class DataCollectionCallback(BaseCallback):

    def __init__(self, save_path, env_name, model_name, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.env_name = env_name
        self.model_name = model_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        # 获取当前环境的 reward 和 done 状态
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        
        # 我们只记录第一个环境的数据
        self.current_rewards += rewards[0]
        self.current_length += 1
        
        if dones[0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            
            # 每收集 10 个 episode 打印一次日志
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else self.current_rewards
                print(f"[{self.model_name}] Episode {len(self.episode_rewards)}: Reward={self.current_rewards:.2f}, Avg10={recent_avg:.2f}, Length={self.current_length}", flush=True)
            
            self.current_rewards = 0.0
            self.current_length = 0
        return True

    def _on_training_end(self) -> None:
        # 训练结束时保存数据到文件并生成图表
        os.makedirs(self.save_path, exist_ok=True)
        
        # 保存原始数据
        save_file = os.path.join(self.save_path, "training_data.npz")
        np.savez(save_file, rewards=self.episode_rewards, lengths=self.episode_lengths)
        
        # 生成折线图
        self.plot_training_progress()
        print(f"训练数据已保存到: {save_file}", flush=True)

    def plot_training_progress(self):
        """生成训练进度的折线图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
            
            if not self.episode_rewards:
                return
            
            plt.figure(figsize=(12, 8))
            
            # 创建两个子图
            plt.subplot(2, 1, 1)
            episodes = range(1, len(self.episode_rewards) + 1)
            plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='原始奖励')
            
            # 添加移动平均线（如果数据足够多）
            if len(self.episode_rewards) >= 10:
                window_size = min(20, len(self.episode_rewards) // 5)
                moving_avg = []
                for i in range(len(self.episode_rewards)):
                    start_idx = max(0, i - window_size + 1)
                    moving_avg.append(np.mean(self.episode_rewards[start_idx:i+1]))
                plt.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'移动平均({window_size})')
            
            plt.title(f'{self.env_name} - {self.model_name} 训练奖励曲线')
            plt.xlabel('Episode')
            plt.ylabel('奖励')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 第二个子图：Episode长度
            plt.subplot(2, 1, 2)
            plt.plot(episodes, self.episode_lengths, 'g-', alpha=0.7, label='Episode长度')
            plt.title(f'{self.env_name} - {self.model_name} Episode长度变化')
            plt.xlabel('Episode')
            plt.ylabel('步数')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            plot_file = os.path.join(self.save_path, f"{self.env_name}_{self.model_name}_training_plot.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"训练曲线图已保存到: {plot_file}", flush=True)
            
        except ImportError:
            print("警告: 无法导入 matplotlib，跳过图表生成。可通过 pip install matplotlib 安装。", flush=True)
        except Exception as e:
            print(f"生成图表时出错: {e}", flush=True)

def make_single_env(env_name):
    """创建单个环境，应用必要的包装器"""
    def _init():
        env = gym.make(env_name)
        # 如果是 MountainCar，应用自定义奖励包装器
        if env_name == "MountainCar-v0":
            env = MountainCarRewardWrapper(env)
        return env
    return _init

test_envs = [
            "MountainCar-v0",         
            "LunarLander-v3",          
            "ALE/Breakout-v5", 
             ]

# 定义训练模型
test_models = [PPO, A2C, DQN, SAC, TD3, DDPG]

def check_compatibility(model_class, env)->tuple[bool, str]:
    action_space = env.action_space
    model_name = model_class.__name__
    #  检查离散动作空间兼容性
    if isinstance(action_space, gym.spaces.Discrete):
        if model_name in ["SAC", "TD3", "DDPG"]:
            return False, f"{model_name} 不支持离散动作空间 (Discrete)"
    # 检查连续动作空间兼容性 (Box)
    elif isinstance(action_space, gym.spaces.Box):
        if model_name in ["DQN"]:
            return False, f"{model_name} 不支持连续动作空间 (Box)"
    return True, "兼容"

def train_and_evaluate(model_class, env_name):
    try:
        # 注册Atari环境
        if env_name.startswith("ALE/"):
            gym.register_envs(ale_py)
        
        # 先创建一个临时环境以检查动作空间
        temp_env = gym.make(env_name)
        
        # 检查兼容性
        is_compatible, reason = check_compatibility(model_class, temp_env)
        if not is_compatible:
            print(f"跳过: {env_name} + {model_class.__name__} -> 原因: {reason}", flush=True)
            temp_env.close()
            return
        
        temp_env.close()
        

        n_envs = 8
        print(f"正在创建 {n_envs} 个并行环境: {env_name}", flush=True)
        env = SubprocVecEnv([make_single_env(env_name) for _ in range(n_envs)])

        callbacks = []

        # 评估环境使用单个环境
        eval_env = make_single_env(env_name)()
        # 保存路径：./logs/环境名_模型名/
        save_path = f"./logs/{env_name}_{model_class.__name__}"
        os.makedirs(save_path, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=5000, # 每 5000 步评估一次
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # 自定义 Callback (用于收集自定义数据)
        # 传入环境名和模型名以便更好地标识数据
        callbacks.append(DataCollectionCallback(save_path=save_path
                                                , env_name=env_name
                                                , model_name=model_class.__name__
                                                , verbose=1))

        print(f"正在训练环境: {env_name} 使用模型: {model_class.__name__}（{n_envs}个并行环境）", flush=True)
        # 获取维度
        obs_shape = env.observation_space.shape
        # 做区别和其他维度判断
        policy_type = 'CnnPolicy' if len(obs_shape) >= 2 else 'MlpPolicy'
        # 针对 MountainCar 的特殊优化参数 
        model_kwargs = { "learning_rate": 0.0005
                        , "gamma": 0.98} if env_name == "MountainCar-v0" else {}
    
        model = model_class(policy_type, env
                            , verbose=1
                            , device="cuda"
                            , tensorboard_log="./tensorboard_logsFinal_Plus/"
                            , **model_kwargs)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        print(f"当前不训练-{env_name} - {model_class.__name__} - 期望: {mean_reward:.1f}", flush=True)    
        total_steps = 2500000        # 开始训练250万步
        model.learn(total_timesteps=total_steps
                    , progress_bar=True, tb_log_name=f"{env_name}_{model_class.__name__}"
                    , callback=callbacks)
        # 训练后再评估
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"当前训练后-{env_name} - {model_class.__name__} - 期望: {mean_reward:.1f}", flush=True)
        
        # 保存最终模型
        model_save_path = os.path.join(save_path, f"final_model")
        model.save(model_save_path)
        print(f"最终模型已保存到: {model_save_path}", flush=True)
        
        env.close()
        eval_env.close()
    except Exception as e:
        print(f"错误显示： {model_class.__name__} 在 {env_name}: {e}", flush=True)

if __name__ == "__main__":
    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     futures = []
    #     for env_name in test_envs:
    #         for model_class in test_models:
    #             futures.append(executor.submit(train_and_evaluate, model_class, env_name))
    #     for future in futures:
    #         future.result()
    for env_name in test_envs:
        for model_class in test_models:
            train_and_evaluate(model_class, env_name)