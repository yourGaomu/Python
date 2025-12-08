import sys

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

sys.modules.setdefault("gym", gym)


checkpoint = load_from_hub("user-l/ppo-LunarLander-v3", "ppo-LunarLander-v3.zip")

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
model = PPO.load(checkpoint, custom_objects=custom_objects)

env = gym.make("LunarLander-v3", render_mode="human")

print("Evaluating model")
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=20,
    deterministic=True,
)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Start a new episode
obs = env.reset()

try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
except KeyboardInterrupt:
    pass
