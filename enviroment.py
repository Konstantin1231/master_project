
# Step 1: Set up the environment.

"""
Classical way to interact with the environment:
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info (we just sample a random action )
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
"""




def initialize_env(env):
    observation, info = env.reset()
    return observation, info

def run_env_step(env, action, random_action=False):
    if random_action :
        action = env.action_space.sample()  # agent policy that uses the observation and info (we just sample a random action )
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info

def env_reset(env):
    observation, info = env.reset()
    return observation, info

def close_env(env):
    env.close()
