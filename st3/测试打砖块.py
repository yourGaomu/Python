import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np
import math
import ale_py

# 注册Atari环境
gym.register_envs(ale_py)

def make_env():
    """创建单个Breakout环境"""
    def _init():
        env = gym.make('ALE/Breakout-v5')
        return env
    return _init

def main():
    n_envs = 4
    print(f"创建 {n_envs} 个并行环境进行训练...")
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    
    print(f"环境创建完成，开始训练...")
    model = PPO("CnnPolicy", env, device="cuda", learning_rate=0.0003, gamma=0.99, n_steps=2048, batch_size=64, verbose=1)  # PPO参数优化
    model.learn(total_timesteps=1000000, progress_bar=True)  
    
    # 保存模型
    model.save("breakout_ppo_model")
    print("模型已保存为 breakout_ppo_model")
    
    env.close()
    
    # 测试模型
    test_model(model)


def test_model(model):
    env = gym.make('ALE/Breakout-v5', render_mode='human') 
    obs, _ = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    steps = 0

    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps > 10000: 
            break

    if total_reward > 300:
        performance = "优秀表现!"
    elif total_reward > 100:
        performance = "良好表现"
    elif total_reward > 0:
        performance = "一般表现"
    else:
        performance = "需要改进"
    
    print(f'Total Reward: {total_reward:.2f}, Steps: {steps}, Performance: {performance}')
    env.close()


if __name__ == "__main__":
    main()