import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Set the device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# Use more parallel environments
env = make_atari_env("PongNoFrameskip-v4", n_envs=32, seed=0)
env = VecFrameStack(env, n_stack=8)

# Set the number of threads for PyTorch operations
torch.set_num_threads(8)

# Use PPO with adjusted parameters
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log="./tensorboard_logs/",
    learning_rate=3e-4,
    batch_size=512,
    n_steps=2048,
)

# Train for a higher number of timesteps to ensure the agent learns effectively
model.learn(total_timesteps=10_000_000, progress_bar=True)
model.save("ppo_pong")

# Track the training progress using TensorBoard
# Run this command in a separate terminal to visualize: tensorboard --logdir=./tensorboard_logs/
