"""
        RUN EPISODES (NON TD)
"""
import random

from enviroment import *
import numpy as np


# Collect episodes using the current policy.
def run_episodes_mtr(agent, env, n_episodes=1, render=False):
    # Initialize the run counter and an empty list to store episodes
    run = 0
    episodes = []
    obs_dim = agent.ObsSpaceDim
    horizon = agent.horizon
    # Run episodes until the specified number of episodes is reached
    while run < n_episodes:
        # Initialize variables for each episode
        episode = []
        state, _ = initialize_env(env, obs_dim=obs_dim)
        if render:
            env.render()
        step = 1
        terminated = False
        # Run each episode until it is terminated or the horizon is reached
        while not terminated and step <= horizon:
            input_vector = np.zeros(agent.ObsSpaceDim + 1)
            input_vector[:agent.ObsSpaceDim] = state
            input_vector[-1] = (agent.horizon - step) / agent.horizon
            # Initialize the input vector and set its elements
            action, probs, action_value = agent.select_action(input_vector)
            # Execute the sampled action in the environment and observe the next state, reward, and termination signal
            next_state, reward, terminated, truncated, info, _ = run_env_step(env, action=action_value,
                                                                              random_action=False, obs_dim=obs_dim)
            # Custom reward
            reward_new = custom_reward(state, reward, terminated, game_name=agent.game_name)
            # Append the experience tuple to the episode
            episode.append((input_vector, step, action, reward_new, probs[action].detach().numpy()))
            # Update the current state and time step
            if render:
                env.render()
            state = next_state
            step += 1
        # Append the completed episode to the list of episodes
        episodes.append(episode)
        run += 1

    # Return the list of episodes
    return episodes


# Collect episodes using the current policy.
def run_episodes(agent, env, n_episodes=1, epsilon=0, render=False):
    run = 0
    episodes = []
    obs_dim = agent.ObsSpaceDim
    while (run < n_episodes):
        episode = []
        truncated = False
        terminated = False
        state, _ = initialize_env(env, obs_dim=obs_dim)
        if render:
            env.render()
        step = 1
        while not terminated and step <= agent.horizon:
            if agent.name in ["MtrNet", "ReinMtrNet", "OriginalMtr"]:
                action, probs, action_value = agent.select_action(state, step)
            else:
                action, probs, action_value = agent.select_action(state)
            if random.uniform(0, 1) < epsilon:
                next_state, reward, terminated, truncated, info, action = run_env_step(env, action=action_value,
                                                                                       random_action=True,
                                                                                       obs_dim=obs_dim)
            else:
                next_state, reward, terminated, truncated, info, _ = run_env_step(env, action=action_value,
                                                                                  random_action=False, obs_dim=obs_dim)
            reward_new = custom_reward(state, reward, terminated, game_name=agent.game_name)
            episode.append((state, step, action, reward_new, probs[action].detach().numpy()))
            if render:
                env.render()
            state = next_state
            step += 1
        episodes.append(episode)

        run += 1
    return episodes


def compute_decay(tau_end, tau_init, ngames, patience):
    return (tau_end / tau_init) ** (patience / ngames)

def train_agent(agent, env, num_epoches, n_episodes, tau_end = 0.2, lr_end = 1e-07, patience = 100, clip_grad=False):
    lr_init = agent.lr
    env_reset(env)
    # list to store loss/rewards
    loss_list = []
    # Train for a number of epochs
    print(f"beta full = {agent.beta}")
    for epoch in range(num_epoches):
        # Collect episodes
        if agent.name == "MTR":
            episodes = run_episodes_mtr(agent, env, n_episodes=n_episodes)
        else:
            episodes = run_episodes(agent, env, n_episodes=n_episodes)
        # Update the policy based on the episodes
        loss_list.append(agent.train(episodes, clip_grad=False))
        if epoch % patience == 0:
            agent.tau = agent.tau * compute_decay(tau_end, agent.tau, num_epoches, patience)
            agent.lr = agent.lr * compute_decay(lr_end, agent.lr, num_epoches, patience)
            agent.set_optimazer()
        if epoch % 20 == 0:
            print(f"Epoch {epoch + 1}")
            print(f"Learning rate {agent.lr * 1000} * 10^{-3}")
            print(f"Reward: {loss_list[-1]}")
        env_reset(env)
    agent.lr = lr_init
    close_env(env)
    return loss_list
