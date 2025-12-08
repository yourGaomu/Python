import gymnasium as gym
from stable_baselines3 import DQN, PPO
import os
import numpy as np
import math

def main():
    env = gym.make('LunarLander-v3')  # 创建环境
    model = PPO("MlpPolicy", device="cpu", env=env, learning_rate=0.0003, gamma=0.99, n_steps=2048, batch_size=64, verbose=1)  # PPO参数优化
    model.learn(total_timesteps=500000,progress_bar=True)  # 训练步数10万
    # # 加载模型
    # model = DQN.load("dqn_mountaincar_tuned", env=env)
    test_model(model)  # 测试模型


def test_model(model):
    env = gym.make('LunarLander-v3', render_mode='human') 
    obs, _ = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    steps = 0

    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps > 1000:  # 防止无限循环，LunarLander通常需要更多步数
            break

    legs_contact = obs[6] == 1.0 and obs[7] == 1.0 
    success = "成功着陆!" if terminated and legs_contact and total_reward > 0 else "着陆失败"
    print(f'Total Reward: {total_reward:.2f}, Steps: {steps}, {success}')
    env.close()


if __name__ == "__main__":
    main()