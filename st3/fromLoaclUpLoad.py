from stable_baselines3 import SAC
import gymnasium as gym

model_path = "models/ant-v5-sac-medium.zip"
model = SAC.load(model_path)

env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset()
for _ in range(1000000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()