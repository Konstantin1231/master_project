"""
        RUN EPISODES (NON TD)
"""
import random

from enviroment import *
import numpy as np
# Collect episodes using the current policy.
def run_episodes_mtr(agent, env, n_episodes=1, game="Lake"):
    # Initialize the run counter and an empty list to store episodes
    run = 0
    episodes = []
    obs_dim = agent.ObsSpaceDim
    horizon = agent.horizon
    # Run episodes until the specified number of episodes is reached
    while (run < n_episodes):
        # Initialize variables for each episode
        episode = []
        state, _ = initialize_env(env, obs_dim=obs_dim)
        step = 0
        terminated = False
        # Run each episode until it is terminated or the horizon is reached
        while (not terminated and step < horizon):
            input_vector = np.zeros(agent.ObsSpaceDim + 1)
            input_vector[:agent.ObsSpaceDim] = state
            input_vector[-1] = (agent.horizon - step) / agent.horizon
            # Initialize the input vector and set its elements
            action, probs, action_value = agent.select_action(input_vector)
            # Execute the sampled action in the environment and observe the next state, reward, and termination signal
            next_state, reward, terminated, truncated, info, _ = run_env_step(env, action=action_value, random_action=False, obs_dim=obs_dim)
            # Custom reward
            reward_new = custom_reward(state, reward, terminated, game_name=game)
            # Append the experience tuple to the episode
            episode.append((input_vector, step, action, reward_new, probs[action].detach().numpy()))
            # Update the current state and time step
            state = next_state
            step += 1
        # Append the completed episode to the list of episodes
        episodes.append(episode)
        run += 1

    # Return the list of episodes
    return episodes


# Collect episodes using the current policy.
def run_episodes(agent, env, n_episodes=1, game="Lake", epsilon = 0):
    run = 0
    episodes = []
    obs_dim = agent.ObsSpaceDim
    while (run < n_episodes):
        episode = []
        truncated = False
        terminated = False
        state, _ = initialize_env(env, obs_dim=obs_dim)
        step = 1
        while (not terminated and step < agent.horizon):
            action, probs, action_value = agent.select_action(state)
            if random.uniform(0,1) < epsilon:
                next_state, reward, terminated, truncated, info, action = run_env_step(env, action=action_value,
                                                                               random_action=True , obs_dim=obs_dim)
            else:
                next_state, reward, terminated, truncated, info, _ = run_env_step(env, action=action_value, random_action=False,obs_dim=obs_dim)
            reward_new = custom_reward(state, reward, terminated, game_name=game)
            episode.append((state, step, action, reward_new, probs[action].detach().numpy() ))
            state = next_state
            step += 1
        episodes.append(episode)

        run += 1
    return episodes
