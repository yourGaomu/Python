import gymnasium as gym
from stable_baselines3 import DQN, PPO
import os
import numpy as np
import math

class MountainCarRewardWrapper(gym.Wrapper):
    """
    自定义奖励包装器：根据位置和速度给予更密集的奖励，加速收敛
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


def main():
    env = gym.make('MountainCar-v0')  # 创建环境
    env = MountainCarRewardWrapper(env)  # 使用自定义奖励包装器
    model = PPO("MlpPolicy", env, learning_rate=0.0003, gamma=0.99, n_steps=2048, batch_size=64, verbose=1)  # PPO参数优化
    model.learn(total_timesteps=200000,progress_bar=True)  # 增加训练步数到20万
    # # 加载模型
    # model = DQN.load("dqn_mountaincar_tuned", env=env)
    test_model(model)  # 测试模型


def test_model(model):
    env = gym.make('MountainCar-v0', render_mode='human')  # 可视化只能在初始化时指定
    obs, _ = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    steps = 0

    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps > 500:  # 防止无限循环
            break

    success = "成功爬上山顶!" if terminated and obs[0] >= 0.5 else "未能爬上山顶"
    print(f'Total Reward: {total_reward}, Steps: {steps}, {success}')
    env.close()


if __name__ == "__main__":
    main()