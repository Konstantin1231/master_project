import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
from ntk import empirical_ntk_ntk_vps, empirical_ntk_jacobian_contraction
from torch.func import functional_call

class CustomLoss(nn.Module):
    """ to simplify the gradient computation and applying log to the output of the  network"""

    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, outputs: torch.Tensor):
        """
        :param C: we pass parameter C, to be sure that optimizer see it.
        """
        return -torch.sum(torch.log(outputs) * self.C)


class SimpleBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim=8):
        super(SimpleBlock, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, n_outputs),  # Output layer
        )

    def forward(self, x, prev_output=None):
        out = self.Q(x)
        return out


class OriginalMTR(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_dim, n_blocks):
        super(OriginalMTR, self).__init__()
        self.Q = nn.ModuleList([SimpleBlock(n_inputs, n_actions, hidden_dim=hidden_dim) for _ in range(n_blocks)])
        self.softmax = nn.Softmax(dim=-1)
        self.output_size = n_actions
        self.horizon = n_blocks

    def forward(self, x, step, tau=1):
        # Select the neural network based on the step
        Q_i = self.Q[self.horizon - step]
        return self.softmax(Q_i(x) / tau)

    # Initialize weights of NN according to the scheme presented on the  page 102 EPFL_TH9825.pdf
    def ntk_init(self, beta=0.5):
        # beta parameter to control chaos order
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Initialize weights as standard Gaussian RVs
                nn.init.normal_(m.weight, mean=0, std=1)
                # Multiply weights by the given scalar
                m.weight.data *= ((1 - beta ** 2) / m.weight.data.size(1)) ** 0.5
                # Initialize biases as standard Gaussian RVs
                nn.init.normal_(m.bias, mean=0, std=1)
                # Multiply biases by the given scalar
                m.bias.data *= beta

        self.apply(init_weights)
        return

    def value(self, x, step, tau):
        return tau * torch.log((torch.exp(self.forward(x, step, tau) / tau).sum() / self.output_size))

    def ntk(self, x1, x2, block_idx, tau=1, mode="full"):
        """
        Neural Tangent Kernel
        :param mode: either "trace" or "full"
        :param x1, x2: Should have batch dimension
        """
        params = {k: v.detach() for k, v in self.named_parameters()}

        def fnet_single(params, x):
            return functional_call(self, params, (x.unsqueeze(0),block_idx ,tau)).squeeze(0)

        result = empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, mode)
        return result[:, 0]
class OriginalMtrAgent:
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, beta=0.5, learning_rate=1e-3):
        self.horizon = horizon
        self.policy = OriginalMTR(n_inputs, n_outputs, hidden_dim, horizon)
        self.optimizer = [optim.Adam(Q_i.parameters(), lr=learning_rate) for Q_i in self.policy.Q]
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = beta
        self.horizon = horizon
        self.game_name = game_name
        self.name = "OriginalMtr"

    def select_action(self, state, step):
        state_tensor = torch.tensor(state, dtype=torch.float)
        action_prob = self.policy(state_tensor, step, tau=self.tau)
        action = torch.multinomial(action_prob, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_prob, [action_value]
        else:
            return action, action_prob, action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        for episode in episodes:
            reward_list = []
            prob_action_list = []
            for _, _, _, reward, a in reversed(episode):
                total_reward += reward
                reward_list.append(reward)
                prob_action_list.append(a)

            entropy_rewards = np.array(reward_list) - self.tau * np.log(
                self.policy.output_size * np.array(prob_action_list))
            """ 
            # Centre rewards by subtraction of the value function
            last_state, last_step, _, _, _ = episode[-1]
            entropy_rewards[-1] = entropy_rewards[-1] - self.policy.value(torch.FloatTensor(last_state), last_step, self.tau).detach().numpy()
            """
            C = np.cumsum(entropy_rewards)
            C = torch.FloatTensor(C)
            # Policy gradient update
            for state, step, action, _, _ in episode:
                # Enable gradient only for the block related to the current step
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor, step=step, tau=self.tau)

                # Compute the policy gradient loss
                loss = CustomLoss(C[self.horizon - step])
                self.optimizer[self.horizon - step].zero_grad()
                loss(action_probs[0][action]).backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.Q[self.horizon - step].parameters(), clip_value=10)

                self.optimizer[self.horizon - step].step()
        return total_reward / len(episodes)

    def ntk(self, x1, x2, step=1, mode="full", batch=False):
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        if not batch:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
        return self.policy.ntk(x1, x2, step, tau=self.tau, mode=mode)