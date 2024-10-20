import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
)
import os


# Set the device to MPS if available
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

# Use more parallel environments for training
env = make_atari_env("PongNoFrameskip-v4", n_envs=32, seed=0)
env = VecFrameStack(env, n_stack=8)
env = VecTransposeImage(
    env
)  # Ensure training env is wrapped properly for PyTorch compatibility

# Create evaluation environment (single environment, same wrappers as training)
eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=8)
eval_env = VecTransposeImage(eval_env)  # Match the training environment's wrappers

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

# Directory where to save the best model
best_model_save_path = "./best_model/"

# Create the directory if it doesn't exist
os.makedirs(best_model_save_path, exist_ok=True)

print(f"Log directory exists: {os.path.exists('./logs/')}")
print(f"Best model directory exists: {os.path.exists(best_model_save_path)}")

# Callback for saving the best model and early stopping
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=20.0,  # Stop training once reward reaches a satisfactory level
    verbose=1,
)

# Custom evaluation callback to handle tolerance in step counting
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_save_path,
    log_path="./logs/",
    eval_freq=31250,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback,
    verbose=1,
)

callbacks = CallbackList([eval_callback])

# Train for a higher number of timesteps to ensure the agent learns effectively
model.learn(total_timesteps=10_000_000, callback=callbacks, progress_bar=True)

# Save the final model after training
model.save("ppo_pong")

# Track the training progress using TensorBoard
# Run this command in a separate terminal to visualize: tensorboard --logdir=./tensorboard_logs/
