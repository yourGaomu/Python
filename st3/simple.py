import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(10000):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()