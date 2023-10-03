"""
        RUN EPISODES (NON TD)
"""
from enviroment import *
import torch
import numpy as np


# Collect episodes using the current policy.
def run_episodes_mtr(policy, env, n_episodes=1, horizon=100, obs_dim=1, game="Frozen"):
    # Initialize the run counter and an empty list to store episodes
    run = 0
    episodes = []

    # Determine the dimensionality of the observation space and define the input vector size
    ObsSpaceDim = obs_dim
    nInput = ObsSpaceDim + 1

    # Run episodes until the specified number of episodes is reached
    while (run < n_episodes):
        # Initialize variables for each episode
        episode = []
        state, _ = initialize_env(env, obs_dim=obs_dim)
        i = 0
        truncated = False
        terminated = False
        # Run each episode until it is terminated or the horizon is reached
        while (not terminated and i < horizon):
            # Initialize the input vector and set its elements
            input_vector = np.zeros(nInput)
            input_vector[:ObsSpaceDim] = state
            input_vector[-1] = (horizon - i) / horizon

            # Convert the input vector to a PyTorch tensor and compute action probabilities using the policy
            input_vector_tensor = torch.tensor(input_vector, dtype=torch.float)
            with torch.no_grad():
                probs = policy(input_vector_tensor)

            # Sample an action according to the computed probabilities
            action = torch.multinomial(probs, num_samples=1).item()

            # Execute the sampled action in the environment and observe the next state, reward, and termination signal
            next_state, reward, terminated, truncated, info = run_env_step(env, action=action, random_action=False)
            reward = custom_reward(state, reward, terminated, game=game)
            # Append the experience tuple to the episode
            episode.append((input_vector, action, reward, probs[action].detach().numpy()))

            # Update the current state and time step
            state = next_state
            i += 1

        # Append the completed episode to the list of episodes
        episodes.append(episode)
        run += 1

    # Return the list of episodes
    return episodes


# Collect episodes using the current policy.
def run_episodes(policy, env, n_episodes=1, limit=1000000, obs_dim=1, game="Frozen", epsilon_greedy=0):
    run = 0
    episodes = []
    while (run < n_episodes):
        episode = []
        truncated = False
        terminated = False
        state, _ = initialize_env(env, obs_dim=obs_dim)
        i = 1
        while (not terminated and i < limit):
            state_tensor = torch.tensor(state, dtype=torch.float)
            probs = policy(state_tensor)
            action = torch.multinomial(probs, num_samples=1).item()
            if np.random.uniform() < epsilon_greedy:
                next_state, reward, terminated, truncated, info = run_env_step(env, action=action, random_action=True,
                                                                               obs_dim=obs_dim)
            else:
                next_state, reward, terminated, truncated, info = run_env_step(env, action=action, random_action=False,
                                                                               obs_dim=obs_dim)
            reward_new = custom_reward(state, reward, terminated, game=game)
            if terminated:
                pass
            episode.append((state, action, reward_new))
            state = next_state
        episodes.append(episode)
        i += 1
        run += 1
    return episodes
