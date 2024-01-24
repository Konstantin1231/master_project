import torch
import random
from enviroment import initialize_env, custom_reward, run_env_step, env_reset, close_env, game_setup
import numpy as np

"""
This section contains functions for following tasks:
- Training loop
- Testing loop
- NTK analysis class: to realize functionality 
"""
def render(agent, env, n_episodes=2):
    if agent.game_name not in ["Toy", "Maze"]:
        env, _, _ = game_setup(agent.game_name, render=True)
    if agent.name == "MTR":
        run_episodes_mtr(agent, env, n_episodes=n_episodes, render=True)
    else:
        run_episodes(agent, env, n_episodes=n_episodes, render=True)

    env.close()

# Collect episodes using the current policy.
def run_episodes_mtr(agent, env, n_episodes=1, render=False, game_name = None):
    # Initialize the run counter and an empty list to store episodes
    run = 0
    episodes = []
    obs_dim = agent.ObsSpaceDim
    horizon = agent.horizon
    # Run episodes until the specified number of episodes is reached
    while run < n_episodes:
        # Initialize variables for each episode
        episode = []
        state, _ = initialize_env(env, obs_dim=obs_dim, game_name=game_name)
        step = 1
        terminated = False
        # Run each episode until it is terminated or the horizon is reached
        while not terminated and step <= horizon:

            input_vector = np.zeros(agent.ObsSpaceDim + 1)
            input_vector[:agent.ObsSpaceDim] = state
            input_vector[-1] = (agent.horizon - step) / agent.horizon
            # Initialize the input vector and set its elements
            action, probs, action_value = agent.select_action(input_vector)
            if render:
                if agent.game_name == "Toy":
                    env.render(agent.horizon, action, probs[action].detach().numpy(), step)
                else:
                    env.render()
            # Execute the sampled action in the environment and observe the next state, reward, and termination signal
            next_state, reward, terminated, truncated, info, _ = run_env_step(env, action=action_value,
                                                                              random_action=False, obs_dim=obs_dim, game_name=game_name)
            # Custom reward
            reward_new = custom_reward(state, reward, terminated, game_name=agent.game_name)
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
def run_episodes(agent, env, n_episodes=1, epsilon=0, render=False, game_name= None):
    if agent.name != "REIN":
        epsilon = 0
    run = 0
    episodes = []
    obs_dim = agent.ObsSpaceDim
    while run < n_episodes:
        episode = []
        truncated = False
        terminated = False
        state, _ = initialize_env(env, obs_dim=obs_dim, game_name=game_name)
        step = 1
        while not terminated and step <= agent.horizon:
            if agent.name in ["REIN"]:
                action, probs, action_value = agent.select_action(state)
            else:
                action, probs, action_value = agent.select_action(state, step)
            if render:
                if agent.game_name == "Toy":
                    env.render(agent.horizon, action, probs[action].detach().numpy(), step)
                else:
                    env.render()
            if random.uniform(0, 1) < epsilon:
                next_state, reward, terminated, truncated, info, action = run_env_step(env,
                                                                                       random_action=True,
                                                                                       obs_dim=obs_dim, game_name=game_name)
            else:
                next_state, reward, terminated, truncated, info, _ = run_env_step(env, action=action_value,
                                                                                  random_action=False, obs_dim=obs_dim, game_name=game_name)
            reward_new = custom_reward(state, reward, terminated, game_name=agent.game_name)
            episode.append((state, step, action, reward_new, probs[action].detach().numpy()))

            state = next_state
            step += 1
        episodes.append(episode)

        run += 1
    return episodes


def compute_decay(tau_end, tau_init, n_episodes, patience):
    return (tau_end / tau_init) ** (patience / n_episodes)


def test_agent(agent, env,game_name, n_episodes = 200, tau = 0.05):
    """
    :param agent: Agent
    :param env: Environment
    :param num_epoches: number of epochs (for training)
    :param n_episodes: number of episodes per epoch
    :param tau_end: Tau at the end of the training
    :param lr_end: Learning rate at the end of the training
    :param patience: Update frequency for agent.tau and agent.lr
    :param clip_grad: clip gradients
    :return: List of total rewards average (average over n_episodes)
    """
    tau_init = agent.tau
    agent.tau = tau
    total_reward = 0
    opt_solution = None
    # list to store loss/rewards
    # Train for a number of epochs
    if game_name == "Toy":
        #generate optimal entropy solution
        env.generate_all_q_stars(agent.horizon, tau=tau)
        v_star = env.v_star[str(agent.horizon)] # take last step-hor V_stars
        if env.rnd_init_state:
            opt_solution = (v_star[0] + v_star[2])/2
        else:
            opt_solution = v_star[0]
    if agent.name == "MTR":
            episodes = run_episodes_mtr(agent, env, n_episodes=n_episodes, game_name=game_name)
    else:
            episodes = run_episodes(agent, env, n_episodes=n_episodes, game_name=game_name)
    # Update the policy based on the episodes
    for episode in episodes:
        # Calculate the returns by iterating through the episode in reverse
        for _, _, _, reward, _ in reversed(episode):
            total_reward += reward
    full_rank = None
    if game_name == "Toy":
        #off testing mode
        env.testing = False
        if agent.name in ["ReinMtrNet", "MtrNet","ShortLongNet","OriginalMtr"]:
            full_rank = False
            ck_analysis = Ntk_analysis(env)
            ck_analysis.full_analysis(agent, mode="ck")
            print(ck_analysis.eigen_values)
            ranks = ck_analysis.rank
            rank = int(np.sum([ranks[str(i+1)] for i in range(agent.horizon)])/agent.horizon)

            if rank == env.n_states:
                full_rank = True
    data = {
        'test': total_reward / len(episodes),
        'full_rank': full_rank,
        'opt_sol': opt_solution,
    }

    close_env(env)
    agent.tau = tau_init
    return total_reward / len(episodes), data



def train_agent(agent, env, num_epoches, game_name, n_episodes, tau_end=0.2, lr_end=1e-07, patience=50,epsilon = 0,
                clip_grad=False, testing=False):
    """
    :param agent: Agent
    :param env: Environment
    :param num_epoches: number of epochs (for training)
    :param n_episodes: number of episodes per epoch
    :param tau_end: Tau at the end of the training
    :param lr_end: Learning rate at the end of the training
    :param patience: Update frequency for agent.tau and agent.lr
    :param clip_grad: clip gradients
    :return: List of total rewards average (average over n_episodes)
    """
    lr_init = agent.lr
    tau_init = agent.tau
    # list to store loss/rewards
    loss_list = []
    entropy_loss_list = []
    # Train for a number of epochs
    print(f" sigma_w = {agent.policy.sigma_w}")
    print(f" sigma_b = {agent.policy.sigma_b}")
    for epoch in range(1,num_epoches+1):
        # Collect episodes
        if agent.name == "MTR":
            episodes = run_episodes_mtr(agent, env, n_episodes=n_episodes, game_name=game_name)
        else:
            episodes = run_episodes(agent, env, n_episodes=n_episodes, epsilon= epsilon, game_name=game_name)
        # Update the policy based on the episodes
        loss = agent.train(episodes, clip_grad=clip_grad)
        loss_list.append(loss[0])
        entropy_loss_list.append(loss[1])
        if epoch % patience == 0:
            agent.tau = agent.tau * compute_decay(tau_end, agent.tau, num_epoches, patience)
            agent.lr = agent.lr * compute_decay(lr_end, agent.lr, num_epoches, patience)
            agent.set_optimazer()
        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1}")
            print(f"tau {agent.tau:.3f}")
            print(f"Learning rate {(agent.lr * 1000):.5f} " + "* 10^-3")
            print(f"Reward: {loss[0]}")
            print(f"Entropy Reward: {loss[1]}")
            if testing:
                test_reward, data_test = test_agent(agent, env, game_name, n_episodes = 80, tau = agent.tau)
                rank = data_test ['full_rank']
                opt_sol = data_test['opt_sol']
                test = data_test['test']
                print(f"Optimal entropy solution: {opt_sol}")
                print(f"CK full rank: {rank}")
                print(f"Test reward: {test}")
            print()
        env_reset(env)
    agent.lr = lr_init
    agent.tau = tau_init
    name = agent.name
    if agent.name in ["ReinMtrNet", "MtrNet","ShortLongNet"]:
        if agent.dynamical:
            name = name + "_" + "Dyn"
        if agent.lightMTR:
            name = name + "_" + "Light"
    data = {
        'game': game_name,
        'agent': name,
        'lr': agent.lr,
        'tau': agent.tau,
        'train': loss_list,
        'train_entropy': entropy_loss_list,
        "total_params": agent.policy.total_number_parameters()
    }
    close_env(env)
    return loss_list, entropy_loss_list, data


def unif_measure(pi1, pi2):
    """
    To measure an error, between two policies
    :param pi1: ndarray of the form [n_satates, n_actions]
    :param pi2: ndarray of the form [n_satates, n_actions]
    :return: The max of the |pi1 - pi2|
    """
    return np.max(pi1 - pi2)


class Ntk_analysis:
    """
        NTK/CK ANALYSIS
    """
    def __init__(self, env):
        self.env = env
        self.rank = {}
        self.eigen_vectors = {}
        self.eigen_values = {}
        self.Q_pi = {}
        self.Q_pi["1"] = env.q_star_1
        self.pi = {}
        self.Q_opt = {}
        self.pi_opt = {}
        self.pi_eval = {}
        self.m = {}


    def ntk_matrix(self, Agent, idx_block):
        """
          Calculating NTK kernel matrix
          :return: Tensor dims (n_inputs * n_actions, n_inputs * n_actions)
          """
        step = Agent.horizon - idx_block  # ntk accept step as input
        dim = self.env.n_states * self.env.n_actions
        ntk_mat = np.zeros((dim, dim))
        for x1 in range(self.env.n_states):
            for x2 in range(self.env.n_states):
                if self.env.one_hot_coded:
                    a = Agent.ntk(self.env.one_hot_decode[x1, :], self.env.one_hot_decode[x2, :], step)[0]
                else:
                    a = Agent.ntk(x1, x2, step)[0]
                ntk_mat[2 * x1, 2 * x2] = a[0, 0]
                ntk_mat[2 * x1 + 1, 2 * x2] = a[1, 0]
                ntk_mat[2 * x1, 2 * x2 + 1] = a[0, 1]
                ntk_mat[2 * x1 + 1, 2 * x2 + 1] = a[1, 1]

        return ntk_mat

    def ck_matrix(self, Agent, idx_block):
        """
          Calculating Conjugate kernel matrix
          :return: Tensor dims (n_inputs * n_actions, n_inputs * n_actions)
          """
        dim = self.env.n_states * self.env.n_actions
        ck_mat = np.zeros((dim, dim))
        for x1 in range(self.env.n_states):
            for x2 in range(self.env.n_states):
                if self.env.one_hot_coded:
                    a = Agent.policy.conjugate_kernel(self.env.one_hot_decode[x1, :], self.env.one_hot_decode[x2, :],
                                                      idx_block)
                else:
                    a = Agent.policy.conjugate_kernel(x1, x2, idx_block)
                ck_mat[2 * x1, 2 * x2] = a
                ck_mat[2 * x1 + 1, 2 * x2] = a
                ck_mat[2 * x1, 2 * x2 + 1] = a
                ck_mat[2 * x1 + 1, 2 * x2 + 1] = a

        return ck_mat

    def ranks(self, mat):
        return np.linalg.matrix_rank(mat)

    def decompose(self, mat):
        return np.linalg.eigh(mat)

    def full_analysis(self, Agent, mode="ntk"):
        for idx_block in range(Agent.horizon):
            if mode == "ntk":
                mat = self.ntk_matrix(Agent, idx_block)
            else:
                mat = self.ck_matrix(Agent, idx_block)
            eigenvalues, eigenvectors = self.decompose(mat)
            self.rank[str(idx_block + 1)] = self.ranks(mat)
            self.eigen_values[str(idx_block + 1)] = eigenvalues[::-1]
            self.eigen_vectors[str(idx_block + 1)] = eigenvectors

    """
      OPTIMALITY CHECK
      """

    def estimate_m(self, Agent, n_rans=500):
        obs_dim = Agent.ObsSpaceDim
        dist = np.zeros((Agent.horizon, self.env.n_states))

        for run in range(n_rans):
            terminated = False
            step = 1
            state, _ = self.env.reset()
            while not terminated and step <= Agent.horizon:
                dist[step - 1, self.env.current_state] = dist[step - 1, self.env.current_state] + 1
                action, probs, action_value = Agent.select_action(state, step)
                next_state, reward, terminated, truncated, info, _ = run_env_step(self.env, action=action_value,
                                                                                  random_action=False,
                                                                                  obs_dim=obs_dim)
                state = next_state
                step += 1

        row_sums = dist.sum(axis=1)
        m = dist / row_sums[:, np.newaxis]
        for step_hor in range(Agent.horizon):
            self.m[str(step_hor + 1)] = m[Agent.horizon - (step_hor + 1), :]
        return self.m

    def generate_agent_policy_matrix(self, Agent):

        for step_horizon in range(Agent.horizon):
            pi_i = np.zeros((self.env.n_states, self.env.n_actions))
            for state in range(self.env.n_states):
                for action in range(self.env.n_actions):
                    state_tensor = torch.tensor(self.env.one_hot_decode[state, :], dtype=torch.float)
                    pi_i[state, :] = Agent.policy(state_tensor, step_horizon, tau=Agent.tau).detach().numpy()
            self.pi[str(step_horizon + 1)] = pi_i
        return self.pi

    def generate_Q_pi(self, policy, Agent, step_hor=2):
        q_previous = self.Q_pi[str(step_hor - 1)]
        q_next = np.zeros((self.env.n_states, self.env.n_actions))
        for state in range(self.env.n_states):
            for action in range(self.env.n_actions):
                next_state = (state + (action + 1)) % self.env.n_states
                q_next[state, action] = self.env.q_star_1[state, action] + np.sum(
                    policy[str(step_hor - 1)][next_state, :] * (q_previous[next_state, :] - Agent.tau * np.log(
                        self.env.n_actions * policy[str(step_hor - 1)][next_state, :])))
        self.Q_pi[str(step_hor)] = q_next
        if step_hor == Agent.horizon:
            print(f"Calculated until horizon {step_hor}")
            return self.Q_pi
        else:
            return self.generate_Q_pi(policy=policy, Agent=Agent, step_hor=step_hor + 1)

    def projection(self, Q, basis, pi, m):
        """
          computing projection of Q on the given basis, w.r.t agent policy distribution
          :param Q: Target function to project
          :return: Proj(Q)
          """

        def inner_product(Q, e_i):
            """
                :param e_i: basis vector: e_i = [(s_0, a_0), (s_0, a_1), (s_1, a_0), ...]
                :param pi: policy matrix of the current horizon: ndarray (s_states, n_actions)
                :param m:  state distribution of current horizon step: ndarray (n_states, .)
                :return:   ndarray scalar
                """
            coef = 0
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    coef += m[s] * pi[s, a] * Q[s, a] * e_i[2 * s + a]

            return coef

        result = np.sum(inner_product(Q, e_i) * e_i for e_i in basis)
        # making good format
        Q_proj = np.zeros((self.env.n_states, self.env.n_actions))
        for s in range(self.env.n_states):
            Q_proj[s, 0] = result[2 * s]
            Q_proj[s, 1] = result[2 * s + 1]

        return Q_proj

    def genarate_optimal_Q(self, Agent, basis=None):
        if basis is None:
            basis = {}
            for step_hor in range(Agent.horizon):
                basis[str(step_hor + 1)] = [self.eigen_vectors[str(step_hor + 1)][:, i] for i in
                                            range(self.eigen_vectors[str(step_hor + 1)].shape[1])]
        if type(basis) is int:
            limit = basis
            basis = {}
            for step_hor in range(Agent.horizon):
                selected_eigen_vectors = self.eigen_vectors[str(step_hor + 1)][:, -limit:]
                basis[str(step_hor + 1)] = [selected_eigen_vectors[:, i] for i in
                                            range(selected_eigen_vectors.shape[1])]

        m = self.estimate_m(Agent)
        pi = self.generate_agent_policy_matrix(Agent)
        Q_pi = self.generate_Q_pi(pi, Agent)
        if Agent.name == "OriginalMtr":
            for step_hor in range(Agent.horizon):
                Q_target = Q_pi[str(step_hor + 1)]
                Q_proj = self.projection(Q_target, basis[str(step_hor + 1)], pi[str(step_hor + 1)], m[str(step_hor + 1)])
                self.Q_opt[str(step_hor + 1)] = Q_proj
        else:
            Q_old_target = 0
            for step_hor in range(Agent.horizon):
                Q_target = Q_pi[str(step_hor + 1)] - Q_old_target
                Q_proj = self.projection(Q_target, basis[str(step_hor + 1)], pi[str(step_hor + 1)], m[str(step_hor + 1)])
                self.Q_opt[str(step_hor + 1)] = Q_old_target + Q_proj
                Q_old_target = pi[str(step_hor + 1)]
            """       
            Q_old_target = 0
            for step_hor in range(Agent.horizon):
                Q_target = Q_pi[str(step_hor + 1)] - Q_old_target
                Q_proj = self.projection(Q_target, basis[str(step_hor + 1)], pi[str(step_hor + 1)], m[str(step_hor + 1)])
                self.Q_opt[str(step_hor + 1)] = Q_old_target + Q_proj
                Q_old_target = self.Q_opt[str(step_hor + 1)]"""


        self.generate_optimal_policy_matrix(Agent)

        return self.Q_opt

    def generate_optimal_policy_matrix(self, Agent, Q_opt=None):
        if Q_opt is None:
            Q_opt = self.Q_opt
        for step_hor in range(Agent.horizon):
            self.pi_opt[str(step_hor + 1)] = self.env.pi_(Q_opt[str(step_hor + 1)], Agent.tau)
        return self.pi_opt

    def policy_eval(self, pi_agent=None, pi_opt=None):
        if pi_agent is None:
            pi_agent = self.pi
        if pi_opt is None:
            pi_opt = self.pi_opt
        for step_hor in range(len(pi_opt.keys())):
            error = [self.m[str(step_hor + 1)][s] * (
                np.abs(pi_agent[str(step_hor + 1)][s, 0] - pi_opt[str(step_hor + 1)][s, 0])) for s in
                     range(self.env.n_states)]
            self.pi_eval[str(step_hor + 1)] = {"overall": np.sum(error),"max": np.max(error), "mean": np.mean(error)}
        return self.pi_eval




