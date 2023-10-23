import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
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
    def __init__(self, n_inputs, n_outputs, hidden_dim=8):
        super(SimpleBlock, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, n_outputs),  # Output layer
        )

    def forward(self, x, prev_output=None):
        out = self.Q(x)
        if prev_output is not None:
            out += prev_output
        return out


class MtrNet(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_dim, n_blocks):
        super(MtrNet, self).__init__()
        # Using _modules to set the block with a particular name
        # Create an OrderedDict to hold the blocks
       # self.blocks = nn.ModuleDict({
       #      f'block_{i+1}': SimpleBlock(n_inputs, n_actions, hidden_dim=hidden_dim)
       #     for i in range(n_blocks)
       # })
        self.blocks = nn.ModuleList([SimpleBlock(n_inputs, n_actions, hidden_dim=hidden_dim) for _ in range(n_blocks)])
        self.softmax = nn.Softmax(dim=-1)
        self.output_size = n_actions

    def forward(self, x, tau=1, softmax=True):
        i = 0
        prev_output = None
        softmax_outputs = []
        Q_outputs = []
        #for name, block in self.named_children():
        for block in self.blocks:
            prev_output = block(x, prev_output)  # passing the original input and the output from the previous block
            Q_outputs.append(prev_output)
            softmax_outputs.append(self.softmax(prev_output / tau))
            i += 1
        if softmax:
            return softmax_outputs
        else:
            return Q_outputs

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

    def value(self, x, tau):
        return tau * torch.log((torch.exp(self.forward(x, tau)[0] / tau).sum() / self.output_size))

    def ntk(self, x1, x2, block_idx, tau, mode="full", softmax=False, show_dim_jac=False):
        """
        Neural Tangent Kernel
        :param softmax: True if we want to apply softmax ob the output(preferences)
        :param mode: either "trace" or "full"
        :param x1, x2: Should have batch dimension
        """
        params = {}
        for name, param in self.named_parameters():
            if f"blocks.{block_idx}" in name:
                params[name] = param.detach()

        # params = {k: v.detach() for k, v in self.named_parameters()}

        def fnet_single(params, x):
            return functional_call(self, params, (x.unsqueeze(0), tau, softmax))[block_idx].squeeze(0)

        result = empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, mode, show_dim_jac=show_dim_jac)
        return result


    def count_parameters_in_block(self, block_idx):
        total_params = 0
        for name, parameter in self.named_parameters():
            if f"blocks.{block_idx}" in name:
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

    """ Methods COPIED FROM Hanveiga """
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


class ReinforceMtrNetAgent:
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, beta=0.5, learning_rate=1e-3):
        self.horizon = horizon
        self.policy = MtrNet(n_inputs, n_outputs, hidden_dim, horizon)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = beta
        self.lr = learning_rate
        self.horizon = horizon
        self.game_name = game_name
        self.name = "ReinMtrNet"

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.lr = lr

    def select_action(self, state, step):
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            softmax_outputs = self.policy(state_tensor, tau=self.tau)
        block_idx = self.horizon - step  # step starts from 1
        action_probs = softmax_outputs[block_idx]  # Using the softmax output of specific block
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_probs.to("cpu"), [action_value]
        else:
            return action, action_probs.to("cpu"), action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        # Disable gradient computation for all parameters
        for param in self.policy.parameters():
            param.requires_grad = False
        for episode in episodes:
            # Calculate the returns for each step in the trajectory
            returns = []
            G = 0

            for _, _, _, reward, _ in reversed(episode):
                total_reward += reward
                G = reward + gamma * G
                returns.insert(0, G)

            returns = torch.FloatTensor(returns)
            # Policy gradient update
            for state, step, action, _, _ in episode:

                block_idx = self.horizon - step
                # Disable gradient for all blocks

                # Enable gradient only for the block related to the current step
                for param in self.policy.blocks[block_idx].parameters():
                    param.requires_grad = True


                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                softmax_outputs = self.policy(state_tensor, tau=self.tau)
                action_probs = softmax_outputs[block_idx]

                # Negative log likelihood of the taken action
                loss = -torch.log(action_probs[0][action]) * returns[- step]
                self.optimizer.zero_grad()
                loss.backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)

                self.optimizer.step()

                for param in self.policy.blocks[block_idx].parameters():
                    param.requires_grad = False


        return total_reward / len(episodes)


    def ntk(self, x1, x2, step, mode="full", batch=False, softmax=False, show_dim_jac=False):
        """
        NTK (NEURAL TANGENT KERNEL)
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


class MtrNetAgent:
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, beta=0.5, learning_rate=1e-3):
        self.horizon = horizon
        self.policy = MtrNet(n_inputs, n_outputs, hidden_dim, horizon)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = beta
        self.lr = learning_rate
        self.horizon = horizon
        self.game_name = game_name
        self.name = "MtrNet"

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.lr = lr

    def select_action(self, state, step):
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            softmax_outputs = self.policy(state_tensor, tau=self.tau)
        block_idx = self.horizon - step  # step starts from 1
        action_probs = softmax_outputs[block_idx]  # Using the softmax output of specific block
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_probs.to("cpu"), [action_value]
        else:
            return action, action_probs.to("cpu"), action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        # Disable gradient computation for all parameters
        for param in self.policy.parameters():
            param.requires_grad = False

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

            # Centre rewards by subtraction of the value function
            # last_state, _, _, _, _ = episode[-1]
            # entropy_rewards[-1] = entropy_rewards[-1] - self.policy.value(torch.FloatTensor(last_state), self.tau).detach().numpy()
            C = np.cumsum(entropy_rewards[::-1])
            C = torch.FloatTensor(C)
            # Policy gradient update
            for state, step, action, _, _ in episode:
                block_idx = self.horizon - step

                # Enable gradient only for the block related to the current step
                for param in self.policy.blocks[block_idx].parameters():
                    param.requires_grad = True

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                softmax_outputs = self.policy(state_tensor, tau=self.tau)
                action_probs = softmax_outputs[block_idx]

                # Compute the policy gradient loss
                loss = CustomLoss(C[block_idx])
                self.optimizer.zero_grad()
                loss(action_probs[0][action]).backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)

                self.optimizer.step()

                for param in self.policy.blocks[block_idx].parameters():
                    param.requires_grad = False

        return total_reward / len(episodes)

    def ntk(self, x1, x2, step, mode="full", batch=False, softmax=False, show_dim_jac=False):
        """
        NTK (NEURAL TANGENT KERNEL)
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


