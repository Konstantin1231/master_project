import gymnasium as gym
from maze_env import CustomMazeEnv
from toy import *


def initialize_env(env, obs_dim=1):
    observation, info = env.reset()
    if obs_dim == 1:
        return [observation], info
    else:
        return observation, info


def run_env_step(env, action=1, random_action=False, obs_dim=2):
    """
    To do: add random action to custom environments
    """
    if random_action:
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


def custom_reward(state, reward, terminated, game_name="Lake"):
    if game_name == "Lake":
        if not terminated:
            reward = -1
        elif terminated and reward == 1:
            reward = 100
        else:
            reward = -20
    elif game_name == "Maze":
        pass
    elif game_name == "Car":
        pass
    elif game_name == "Pendulum":
        pass
    elif game_name == "Toy":
        pass
    return reward


def game_setup(game_name, render=False):
    if game_name == "Cart":
        if render == True:
            env = gym.make('CartPole-v1', render_mode="human")
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
            env = gym.make("MountainCar-v0", render_mode="human")
        else:
            env = gym.make("MountainCar-v0")
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = env.action_space.n
    elif game_name == "Lake":
        slippery = False
        if render == True:
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery, render_mode="human")
        else:
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery)
        obs_dim = 1
        action_dim = env.action_space.n
    elif game_name == "Pendulum":
        gravity = 6
        nOutput = 12 # will produce discrete space of action containing nOutput action samy spaced in the interval [-2,2]
        if render == True:
            env = gym.make('Pendulum-v1', g=gravity, render_mode="human")
        else:
            env = gym.make('Pendulum-v1', g=gravity)
        obs, _ = env.reset()
        obs_dim = len(obs)
        action_dim = nOutput
    elif game_name == "Toy":
        alphas = [0.5, 1, -1, 2, 0.2, -0.7, 1, -0.2, 7]
        random_basis = True
        one_hot_coded = True
        env = Toy_env(alphas, random_basis=random_basis, one_hot_coded=one_hot_coded)
        if one_hot_coded:
            obs_dim = env.n_states
        else:
            obs_dim = 1
        action_dim = env.n_actions
    return env, obs_dim, action_dim


from utils import run_episodes, run_episodes_mtr


def render(agent, env,n_episodes=2):
    game_name = agent.game_name
    if game_name in ["Toy", "Maze"]:
        if agent.name == "REIN":
            run_episodes(agent, env, n_episodes=n_episodes, game=game_name, render=True)
        elif agent.name in ["MTR", "MtrNet","OriginalMtr" ]:
            agent.tau = 0.001
            run_episodes_mtr(agent, env, n_episodes=n_episodes, game=game_name, render=True)
    else:
        env, _, _ = game_setup(game_name, render=True)
        if agent.name == "REIN":
            run_episodes(agent, env, n_episodes=n_episodes, game=game_name)
        elif agent.name in ["MTR", "MtrNet","OriginalMtr" ]:
            agent.tau = 0.001
            run_episodes_mtr(agent, env, n_episodes=n_episodes, game=game_name)

    env.close()
