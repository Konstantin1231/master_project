import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Toy_env():
    def __init__(self, alphas=np.array([1, 1, 3, 2, 1]), n_actions=2, random_basis=False, one_hot_coded=False):
        """
        §   Working only with n_actions = 2 !!!
        """
        self.n_actions = n_actions
        self.n_states = len(alphas)
        self.current_state = 0
        self.alphas = alphas
        self.basis = self.generate_basis(random=random_basis)
        self.q_star_1 = self.generate_q_star_1()
        self.q_star_2 = self.generate_q_star_2()
        self.one_hot_coded = one_hot_coded

    def return_random_state(self):
        return np.random.randint(low=0, high=self.n_states - 1)

    def reset(self):
        self.current_state = 0
        if self.one_hot_coded:
            return np.eye(self.n_states)[self.current_state, :], {}
        else:
            return self.current_state, {}

    def step(self, action):
        reward = self.q_star_1[self.current_state, action]

        self.current_state = (self.current_state + action + 1) % self.n_states
        if self.one_hot_coded:
            return np.eye(self.n_states)[self.current_state, :], reward, {}, {}, {}
        else:
            return self.current_state, reward, {}, {}, {}

    def generate_basis(self, random):
        basis = np.zeros((self.n_states, self.n_states, self.n_actions))
        if random:
            Q, R = np.linalg.qr(np.random.rand(self.n_states, self.n_states))
        else:
            Q = np.eye(self.n_states)
        basis[:, :, 0] = Q
        basis[:, :, 1] = -basis[:, :, 0]
        return basis

    def generate_q_star_1(self):
        q_star_1 = np.zeros((self.n_states, self.n_actions))
        for i in range(self.n_states):
            q_star_1 += self.alphas[i] * self.basis[i, :, :]
        return q_star_1

    def generate_q_star_2(self):
        q_2 = np.zeros((self.n_states, self.n_actions))

        for state in range(self.n_states):
            for action in range(self.n_actions):  # offset by -1
                next_state = (state + (action + 1)) % self.n_states
                prefs = self.q_star_1[next_state, :]
                v_star = np.log((np.exp(prefs)).sum() / (self.n_actions))
                q_2[state, action] = self.q_star_1[state, action] + v_star
        return q_2

    def pi_(self, preference):
        """"
        :param preference: Q_i
        :return: Softmax policy based on the Q_i reference
        """
        policy = np.zeros((self.n_states, self.n_actions))
        for i in range(self.n_states):
            policy[i, :] = np.exp(preference[i, :]) / (np.exp(preference[i, 0]) + np.exp(preference[i, 1]))
        return policy

    def render(self):
        fig, ax = plt.subplots()

        # Create a line of circles: each circle represents a state
        for i in range(self.n_states):
            circle = patches.Circle((i * 2.0, 1), 0.5, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(circle)

            # If it's the current state, fill the circle
            if i == self.current_state:
                circle = patches.Circle((i * 2.0, 1), 0.5, linewidth=1, edgecolor='blue', facecolor='blue')
                ax.add_patch(circle)

        reward_1 = self.q_star_1[self.current_state, 0]  # Reward for the first circle (action '1')
        reward_2 = self.q_star_1[self.current_state, 1]  # Reward for the second circle (action '2')

        # Determine the color of the circles based on the reward value
        if reward_1 > reward_2:
            action_circle1 = patches.Circle(((self.current_state + 1) % self.n_states * 2.0, 1), 0.5, linewidth=1,
                                            edgecolor="green",
                                            facecolor='none')
            ax.add_patch(action_circle1)
        else:
            action_circle2 = patches.Circle(((self.current_state + 2) % self.n_states * 2.0, 1), 0.5, linewidth=1,
                                            edgecolor="green", facecolor='none')
            ax.add_patch(action_circle2)

        plt.xlim(-2, self.n_states * 2)
        plt.ylim(-1, 3)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')  # Turn off the axis
        plt.show()

    def close(self):
        self.reset()
        # Closes all the figure windows.
        plt.close('all')
