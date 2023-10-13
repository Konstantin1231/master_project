import gymnasium as gym
from maze_env import CustomMazeEnv
def initialize_env(env, obs_dim = 1):
    observation, info = env.reset()
    if obs_dim == 1:
        return [observation], info
    else:
        return observation, info

def run_env_step(env, action, random_action=False, obs_dim = 1):
    if random_action :
        action = env.action_space.sample()  # agent policy that uses the observation and info (we just sample a random action )
    observation, reward, terminated, truncated, info = env.step(action)
    if obs_dim == 1:
        return [observation], reward, terminated, truncated, info, action
    else:
        return observation, reward, terminated, truncated, info, action

def env_reset(env):
    observation, info = env.reset()
    return observation, info

def close_env(env):
    env.close()

def custom_reward(state, reward, terminated, game_name = "Lake"):
    if game_name == "Lake":
            if not terminated:
                reward = -1
            elif terminated and reward == 1:
                reward = 100
            else:
                reward = -20
    else:
        pass
    return reward

def game_setup(game_name, render= False):
    if game_name == "Cart":
        if render == True:
            env = gym.make('CartPole-v1', render_mode ="human")
        else:
            env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = env.action_space.n
    elif game_name == "Maze":
        env = CustomMazeEnv()
        obs_dim = 2
        action_dim = env.action_space.n
    elif game_name == "Car":
        if render == True:
            env = gym.make("MountainCar-v0", render_mode ="human")
        else:
            env = gym.make("MountainCar-v0")
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = env.action_space.n
    elif game_name == "Lake":
        slippery = False
        if render == True:
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery, render_mode ="human")
        else:
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery)
        obs_dim = 1
        action_dim = env.action_space.n
    elif game_name == "Pendulum":
        gravity = 6
        nOutput = 12
        if render == True:
            env = gym.make('Pendulum-v1', g=gravity, render_mode ="human")
        else:
            env = gym.make('Pendulum-v1', g=gravity)
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim =  nOutput

    return env, obs_dim, action_dim



