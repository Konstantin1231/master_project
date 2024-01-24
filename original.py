import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
from ntk import empirical_ntk_ntk_vps, empirical_ntk_jacobian_contraction
from torch.func import functional_call
import copy
import pickle


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
    """
    Construction of individual blocks
    """
    def __init__(self, n_inputs, n_outputs, hidden_dim=8):
        super(SimpleBlock, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, hidden_dim),  # hidden layer
            nn.ReLU(),  # Activation function
        )
        self.output_layer = nn.Linear(hidden_dim, n_outputs)  # Output layer
    def forward(self, x):
        out = self.Q(x)
        out = self.output_layer(out)
        return out

    def conjugate_kernel(self, x1, x2):
        """
        Conjugate Kernel
        :param x1: input
        :param x2: input
        :return: torch scalar (conjugate_kernel)
        """
        with torch.no_grad():
            a_x1 = self.Q(x1)
            a_x2 = self.Q(x2)
        # Compute the dot product (Conjugate Kernel)
        ck = torch.dot(a_x1.flatten(), a_x2.flatten())
        return ck.numpy()


class OriginalMTR(nn.Module):
    """
    Original Neural Network Constraction
    """
    def __init__(self, n_inputs, n_actions, hidden_dim, n_blocks):
        super(OriginalMTR, self).__init__()
        self.sigma_b = None
        self.sigma_w = None
        self.Q = nn.ModuleList([SimpleBlock(n_inputs, n_actions, hidden_dim=hidden_dim) for _ in range(n_blocks)])
        self.softmax = nn.Softmax(dim=-1)
        self.output_size = n_actions
        self.horizon = n_blocks
        self.ntk_init()

    def forward(self, x, horizen_step, tau=1, softmax=True):
        # Select the neural network based on the step
        Q_i = self.Q[horizen_step]
        if softmax:
            return self.softmax(Q_i(x) / tau)
        else:
            return Q_i(x)

    # Initialize weights of NN according to the scheme presented on the  page 102 EPFL_TH9825.pdf
    def ntk_init(self, sigma_w=np.sqrt(2), sigma_b=0):
        """
        NTK parametrization.
        :param sigma_w: sigma_w * W
        :param sigma_b: sigma_b * bias
        :return: None
        """
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        # beta parameter to control chaos order
        def init_weights(m):

            if isinstance(m, nn.Linear):
                # Initialize weights as standard Gaussian RVs
                nn.init.normal_(m.weight, mean=0, std=1)
                # Multiply weights by the given scalar
                m.weight.data *= sigma_w /(m.weight.data.size(1) ** 0.5 )
                # Initialize biases as standard Gaussian RVs
                nn.init.normal_(m.bias, mean=0, std=1)
                # Multiply biases by the given scalar
                m.bias.data *= sigma_b

        self.apply(init_weights)
        return

    def value(self, x, horizen_step, tau):
        return tau * torch.log((torch.exp(self.forward(x, horizen_step, tau, softmax=False) / tau).sum() / self.output_size))

    def conjugate_kernel(self, x1, x2, block_idx):
        """
        Conjugate kernel by blocks
        :param block_idx: idx of the block, we want to extract features
        block_idx is similar to the horizon step = horizon - step, belong to the interval [0,1,2 ... n-1]
        :return: Conjugate kernel
        """
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        selected_block = self.Q[block_idx]
        return selected_block.conjugate_kernel(x1,x2)

    def ntk(self, x1, x2, block_idx, tau=1, mode="full", softmax=False, show_dim_jac=False):
        """
        Neural Tangent Kernel (NEW version)
        :param mode: either "trace" or "full"
        :param x1, x2: Should have batch dimension
        :return: [ntk (ndarray), Jacobian (list(torch))]
        """

        # params = {k: v.detach() for k, v in self.named_parameters()}
        params = {}
        for name, param in self.named_parameters():
            if f"Q.{block_idx}.Q" in name:
                params[name] = param.detach()

        def fnet_single(params, x):
            return functional_call(self, params, (x.unsqueeze(0), int(block_idx), tau, softmax)).squeeze(0)

        result = empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, mode, show_dim_jac=show_dim_jac)
        return result

    def count_parameters_in_block(self, block_idx):
        total_params = 0
        for name, parameter in self.named_parameters():
            if f"Q.{block_idx}.Q" in name:
                total_params += parameter.numel()  # numel returns the total number of elements in the tensor
        return total_params

    def total_number_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            total_params += parameter.numel()  # numel returns the total number of elements in the tensor
        return total_params


    def save_parameters(self, file_path):
        """
        Save the model parameters to a file using pickle.

        Parameters:
            file_path (str): Path to the file to save the parameters to.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.state_dict(), file)

    def load_parameters(self, file_path):
        """
        Load the model parameters from a file using pickle and set them to the current model.

        Parameters:
            file_path (str): Path to the file containing the saved parameters.
        """
        with open(file_path, 'rb') as file:
            self.load_state_dict(pickle.load(file))

    """ CODE COPIED FROM Hanveiga """

    def norm_param(self):
        # prints norm of parameters per layer
        print("The weights per layer of the NN have norm:")
        for layer in self.state_dict():
            print(torch.norm(self.state_dict()[layer]))

    def rescale_weights(self, gamma=1.0):
        print("Rescalling weights")
        for layer in self.state_dict():
            if len(self.state_dict()[layer].shape) == 2:
                _, input_size = self.state_dict()[layer].shape
                self.state_dict()[layer].mul_(1. / input_size ** gamma)

    def store_weights(self):
        return copy.deepcopy(self.state_dict())

    def compute_change_in_weights(self, old_weights):
        for layer in self.state_dict():
            if len(self.state_dict()[layer].shape) == 2:
                print(torch.norm(old_weights[layer] - self.state_dict()[layer]))

    def restore_weights(self, weights_state_dict):
        for layer in self.state_dict():
            self.state_dict()[layer] = weights_state_dict[layer]


class OriginalMtrAgent:
    """
    Original Agent
    """
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, learning_rate=1e-3):
        self.horizon = horizon
        self.policy = OriginalMTR(n_inputs, n_outputs, hidden_dim, horizon)
        self.optimizer = [optim.Adam(Q_i.parameters(), lr=learning_rate) for Q_i in self.policy.Q]
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = self.policy.sigma_b
        self.lr = learning_rate
        self.horizon = horizon
        self.game_name = game_name
        self.name = "OriginalMtr"

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = [optim.Adam(Q_i.parameters(), lr=self.lr) for Q_i in self.policy.Q]
        else:
            self.optimizer = [optim.Adam(Q_i.parameters(), lr=lr) for Q_i in self.policy.Q]
            self.lr = lr

    def select_action(self, state, step):
        horizon_step = self.horizon - step
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_prob = self.policy(state_tensor, horizon_step, tau=self.tau)
        action = torch.multinomial(action_prob, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_prob, [action_value]
        else:
            return action, action_prob, action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        total_entopy_reward = 0
        for episode in episodes:
            reward_list = []
            prob_action_list = []
            for _, _, _, reward, a in episode:
                total_reward += reward
                reward_list.append(reward)
                prob_action_list.append(a)
            entropy_rewards = np.zeros(self.horizon)
            entropy_rewards[:len(reward_list)] = np.array(reward_list) - self.tau * np.log(
                self.policy.output_size * np.array(prob_action_list))
            """ 
            # Centre rewards by subtraction of the value function
            last_state, last_step, _, _, _ = episode[-1]
            entropy_rewards[-1] = entropy_rewards[-1] - self.policy.value(torch.FloatTensor(last_state), self.horizon - last_step, self.tau).detach().numpy()
            """
            C = np.cumsum(entropy_rewards[::-1])
            total_entopy_reward += C[-1]
            C = torch.FloatTensor(C)
            # Policy gradient update
            for state, step, action, _, _ in episode:
                # Enable gradient only for the block related to the current step
                horizon_step = self.horizon - step
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor, horizon_step, tau=self.tau)

                # Compute the policy gradient loss
                loss = CustomLoss(C[horizon_step])
                self.optimizer[horizon_step].zero_grad()
                loss(action_probs[0][action]).backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.Q[horizon_step].parameters(), clip_value=10)

                self.optimizer[horizon_step].step()
        return total_reward / len(episodes), total_entopy_reward/len(episodes)

    def ntk(self, x1, x2, step, mode="full", batch=False, softmax=False, show_dim_jac=False):
        """
        NTK (NEURAL TANGENT KERNEL)
        :param step: game step (starting from 1)
        :param x1: first input tensor (can be in batch)
        :param x2: second input tensor (can be in batch)
        :param mode: two options: "full", "trace"
        :param batch: True, if you have batch inputs
        :param softmax: True, if you want to apply softmax on the preferences (Q) output.Default is False
        :param show_dim_jac: True, if you wish to check Layers names and dimension used to calculate the Jacobian
        :return: list, where list[0] is ntk tensor, and list[1] is Jacobian. ntk = jac(x1) @ jac^T(x2)
        """
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        if not batch:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
        return self.policy.ntk(x1, x2, self.horizon - step, tau=self.tau, mode=mode, softmax=softmax,
                               show_dim_jac=show_dim_jac)
