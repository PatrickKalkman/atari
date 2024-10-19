from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
model = A2C.load("a2c_pong", env=env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward}, Standard deviation: {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
