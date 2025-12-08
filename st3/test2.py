import gymnasium as gym
from stable_baselines3 import PPO
import torch

class MyWrapper(gym.Wrapper):
    def __init__(self):
        base_env = gym.make('Pendulum-v1', render_mode="human")
        super().__init__(base_env)
        self.env = base_env

    def reset(self, *, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        return state, info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info

# 查看是否支持GPU
print(torch.cuda.is_available())
device = "cpu"
env=MyWrapper()
env.reset()

if torch.cuda.is_available():
    device = "cuda"

model = PPO('MlpPolicy', env, verbose=0, device=device)

# 优先学习再评估,学习的步数可以根据需要调整
model.learn(total_timesteps=30000,progress_bar=True)    

# 去保存这个模型
model.save("models/save")


# 加载模型,并且指定使用GPU，如果不指定device，则默认使用CPU
model = PPO.load("models/save", env=env, device=device)
