import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=32)

model = PPO("CnnPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=10_000_000, progress_bar=True)
model.save("ppo_pong")
