import random

import gymnasium as gym
from maze_env import CustomMazeEnv
from toy import Toy_env
import numpy as np


def env_input_modification(obs, game_name, modify = True):
    """
    One-hot encoded for Lake env.
    Might be used to target other environment as-well.
    When modify = True, we replace standard input with our modified version.
    """
    if modify:
        if game_name == "Lake":
            """
            Only for Lake 4x4
            """
            return np.eye(16)[obs, :]
        else:
            return obs
    else:
        return obs

def initialize_env(env, obs_dim=1, game_name=None):
    """
    We rewrite the common initialization by gym library, to target the case when observable dimension is equal to 1.
    """
    observation, info = env.reset()
    observation = env_input_modification(observation, game_name)
    if obs_dim == 1:
        return [observation], info
    else:
        return observation, info


def run_env_step(env, action=1, random_action=False, obs_dim=2, game_name=None):
    """
    Used to call env.step method with provided action. Again do not simply call env.step since we target the case when observable dimension is equal to 1
    """
    if random_action:
        # for custom environments
        if game_name in ["Toy", "Maze"]:
            actions = [int(i) for i in range(env.n_actions)]
            action = random.choice(actions)
        # for gymnasium envs.
        else:
            action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    observation = env_input_modification(observation, game_name)
    if obs_dim == 1:
        return [observation], reward, terminated, truncated, info, action
    else:
        return observation, reward, terminated, truncated, info, action


def env_reset(env, game_name=None):
    observation, info = env.reset()
    observation = env_input_modification(observation, game_name)
    return observation, info


def close_env(env):
    env.close()


def custom_reward(state, reward, terminated, game_name="Lake"):
    """
    Used to modify environment reward.
    """
    if game_name == "Lake":
        if not terminated:
            reward = -1
        elif terminated and reward == 1:
            reward = 30
        else:
            reward = -5
    elif game_name == "Maze":
        pass
    elif game_name == "Car":
        pass
    elif game_name == "Pendulum":
        pass
    elif game_name == "Toy":
        pass
    elif game_name == "Cart":
        if not terminated:
            reward = 1
        else:
            reward = 0

    return reward


def game_setup(game_name, render=False):
    """
    Used to set up environment's hyperparameters.
    """
    if game_name == "Cart":
        if render == True:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = env.action_space.n

    elif game_name == "Maze":
        size = 5
        one_hot_coded = True
        reward_pos = (4, 4)
        wall_pose = (2, 2)
        reward = 10
        env = CustomMazeEnv(maze_size=size, reward_pos=reward_pos, wall_pose=wall_pose, reward=reward,
                            one_hot_coded=one_hot_coded)
        if one_hot_coded:
            obs_dim = env.grid_size ** 2
        else:
            obs_dim = 2
        action_dim = env.action_space.n

    elif game_name == "Car":
        if render == True:
            env = gym.make("MountainCar-v0", render_mode="human")
        else:
            env = gym.make("MountainCar-v0")
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = env.action_space.n

    elif game_name == "Lake":
        slippery = False
        one_hot_coded = True
        if render == True:
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery, render_mode="human")
        else:
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery)
        if one_hot_coded:
            obs_dim = 16
        else:
            obs_dim = 1
        action_dim = env.action_space.n

    elif game_name == "Pendulum":
        gravity = 6
        nOutput = 12  # will produce discrete space of action containing nOutput action samy spaced in the interval [-2,2]
        if render == True:
            env = gym.make('Pendulum-v1', g=gravity, render_mode="human")
        else:
            env = gym.make('Pendulum-v1', g=gravity)
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = nOutput

    elif game_name == "Toy":
        alphas_master_proj = [0.1, 0.2, -0.1, -0.2,0.1,-0.3,0.8]
        random_basis = False #in Project False
        one_hot_coded = True
        env = Toy_env(alphas_master_proj, random_basis=random_basis, one_hot_coded=one_hot_coded)
        if one_hot_coded:
            obs_dim = env.n_states
        else:
            obs_dim = 1
        action_dim = env.n_actions

    return env, obs_dim, action_dim




