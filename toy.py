import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

class Toy_env():
    def __init__(self, alphas=np.array([1, 1, 3, 2, 1]), n_actions=2, random_basis=False, one_hot_coded=False):
        """
        §   Working only with n_actions = 2 !!!
        """
        self.tau = None
        self.n_actions = n_actions
        self.n_states = len(alphas)
        self.current_state = 0
        self.alphas = alphas
        self.basis = self.generate_basis(random=random_basis)
        self.one_hot_coded = one_hot_coded
        self.q_star = {}
        self.v_star = {}
        self.q_star_1 = self.generate_q_star_1()
        self.q_star[str(1)] = self.q_star_1
        self.generated = False

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

    def generate_all_q_stars(self, horizon, step_hor=2, tau=1):
        q_previous = self.q_star[str(step_hor - 1)]
        q_next = np.zeros((self.n_states, self.n_actions))
        v_previous = np.zeros(self.n_states)
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state = (state + (action + 1)) % self.n_states
                prefs = q_previous[next_state, :]
                v_previous[state] = tau * np.log((np.exp(prefs / tau)).sum() / self.n_actions)
                q_next[state, action] = self.q_star_1[state, action] + v_previous[state]
        self.v_star[str(step_hor - 1)] = v_previous
        self.q_star[str(step_hor)] = q_next
        if step_hor == horizon:
            # calculating last v_star
            self.v_star[str(step_hor)] = np.zeros(self.n_states)
            for state in range(self.n_states):
                self.v_star[str(step_hor)][state] = tau * np.log(
                    (np.exp(q_next[state, :] / tau)).sum() / self.n_actions)
            self.tau = tau
            return print(f"Calculated until horizon {step_hor}")
        else:
            return self.generate_all_q_stars(horizon=horizon, step_hor=step_hor + 1, tau=tau)

    def generate_basis(self, random=False):
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


    def pi_(self, preference):
        """"
        :param preference: Q_i
        :return: Softmax policy based on the Q_i reference
        """
        policy = np.zeros((self.n_states, self.n_actions))
        for i in range(self.n_states):
            policy[i, :] = np.exp(preference[i, :]) / (np.exp(preference[i, 0]) + np.exp(preference[i, 1]))
        return policy

    from matplotlib.patches import FancyArrowPatch

    def render(self, horizon, action, action_prob, step=1):
        fig, ax = plt.subplots()

        circle_radius = 0.5
        text_size = circle_radius * 15  # Adjust this as needed to fit the text inside the circle
        offset = 1
        # Create a line of circles: each circle represents a state
        for i in range(self.n_states):
            circle = patches.Circle((i * 2.0, offset), circle_radius, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(circle)

            # If it's the current state, fill the circle
            if i == self.current_state:
                circle = patches.Circle((i * 2.0, offset), circle_radius, linewidth=1, edgecolor='blue', facecolor='blue')
                ax.add_patch(circle)

        reward_1 = self.q_star[str(horizon - step + 1)][self.current_state, 0]  # Reward for action '1'
        reward_2 = self.q_star[str(horizon - step + 1)][self.current_state, 1]  # Reward for action '2'

        # Determine the color of the circles based on the reward value
        if reward_1 > reward_2:
            if action == 0:
                well_chosen = True
                color = "green"
            else:
                well_chosen = False
                color = "red"
            chosen_reward = reward_1
            non_chosen_reward = reward_2
            chosen_action_circle = patches.Circle(((self.current_state + 1) % self.n_states * 2.0, 1), circle_radius,
                                                  linewidth=1, edgecolor="green", facecolor='none')
            non_chosen_action_circle = patches.Circle(((self.current_state + 2) % self.n_states * 2.0, 1),
                                                      circle_radius,
                                                      linewidth=1, edgecolor="red", facecolor='none')
        else:
            if action == 1:
                well_chosen = True
                color = "green"
            else:
                well_chosen = False
                color = "red"
            chosen_reward = reward_2
            non_chosen_reward = reward_1
            chosen_action_circle = patches.Circle(((self.current_state + 2) % self.n_states * 2.0, 1), circle_radius,
                                                  linewidth=1, edgecolor="green", facecolor='none')
            non_chosen_action_circle = patches.Circle(((self.current_state + 1) % self.n_states * 2.0, 1),
                                                      circle_radius,
                                                      linewidth=1, edgecolor="red", facecolor='none')

        ax.add_patch(chosen_action_circle)
        ax.add_patch(non_chosen_action_circle)
        # Add the reward text inside the chosen and non-chosen action circles
        ax.text(chosen_action_circle.center[0], chosen_action_circle.center[1], f"{chosen_reward:.1f}",
                horizontalalignment='center', verticalalignment='center', color='green', size=text_size)
        ax.text(non_chosen_action_circle.center[0], non_chosen_action_circle.center[1], f"{non_chosen_reward:.1f}",
                horizontalalignment='center', verticalalignment='center', color='red', size=text_size)

        # Draw curved arrows from the top of the current state to the top of the potential next states
        arc_height = 0.7  # This value can be adjusted to change the curvature of the arcs
        arrow_properties = dict(arrowstyle="->", color=color, linewidth=0.6)

        # Chosen action arrow
        chosen_arc = FancyArrowPatch((self.current_state * 2.0, circle_radius + offset),
                                     ((self.current_state + (action + 1)) % self.n_states * 2.0, circle_radius + offset),
                                     connectionstyle=f"arc3,rad=-{arc_height}", **arrow_properties)
        ax.add_patch(chosen_arc)

        ax.text(self.current_state * 2.0, offset, f"{action_prob:.1f}",
                horizontalalignment='center', verticalalignment='center', color=color, size=text_size)

        # Add text in the top left corner with the current state and step horizon
        ax.text(0.01, 0.98, f"Current state: {self.current_state}\nStep horizon: {horizon - step + 1}",
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                color='black', fontsize=text_size)  # Adjust fontsize as needed

        plt.xlim(-2, self.n_states * 2)
        plt.ylim(-0.1, 3)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()

    def close(self):
        self.reset()
        # Closes all the figure windows.
        plt.close('all')
