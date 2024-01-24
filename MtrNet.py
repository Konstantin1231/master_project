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
            nn.Linear(hidden_dim, hidden_dim),  #  # hidden layer
            nn.ReLU(),  # Activation function
        )
        self.output_layer = nn.Linear(hidden_dim, n_outputs) # Output layer

    def forward(self, x, prev_output=None):
        out = self.Q(x)
        out = self.output_layer(out)
        if prev_output is not None:
            out += prev_output
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

class MtrNet(nn.Module):
    """
    MtrNet Neural Network
    """
    def __init__(self, n_inputs, n_actions, hidden_dim, n_blocks, dynamical_layer_param=False):
        super(MtrNet, self).__init__()
        self.sigma_b = None
        self.sigma_w = None
        # dynamical layers
        if dynamical_layer_param:
            self.blocks = nn.ModuleList(
                [SimpleBlock(n_inputs, n_actions,
                             hidden_dim=int(n_actions + (hidden_dim - n_actions) * (n_blocks - i) / n_blocks)) for i in
                 range(n_blocks)])
        else:
            self.blocks = nn.ModuleList(
                [SimpleBlock(n_inputs, n_actions, hidden_dim=hidden_dim) for _ in range(n_blocks)])
        self.softmax = nn.Softmax(dim=-1)
        self.output_size = n_actions
        self.ntk_init()

    def forward(self, x, horizon_step, tau=1, softmax=True):
        prev_output = None
        for block in self.blocks[:horizon_step + 1]:
            prev_output = block(x, prev_output)  # passing the original input and the output from the previous block
        if softmax:
            return self.softmax(prev_output / tau)
        else:
            return prev_output

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
                m.weight.data *= sigma_w / (m.weight.data.size(1) ** 0.5)
                # Initialize biases as standard Gaussian RVs
                nn.init.normal_(m.bias, mean=0, std=1)
                # Multiply biases by the given scalar
                m.bias.data *= sigma_b

        self.apply(init_weights)
        return

    def value(self, x, horizen_step, tau):
        return tau * torch.log(
            (torch.exp(self.forward(x, horizen_step, tau, softmax=False) / tau).sum() / self.output_size))

    def conjugate_kernel(self, x1, x2, block_idx):
        """
        Conjugate kernel by blocks
        :param block_idx: idx of the block, we want to extract features
        block_idx is similar to the horizon step = horizon - step, belong to the interval [0,1,2 ... n-1]
        :return: Conjugate kernel
        """
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        selected_block = self.blocks[block_idx]
        return selected_block.conjugate_kernel(x1,x2)


    def ntk(self, x1, x2, block_idx, tau, mode="full", softmax=False, show_dim_jac=False):
        """
        Neural Tangent Kernel
        :param softmax: True if we want to apply softmax ob the output(preferences)
        :param mode: either "trace" or "full"
        :param x1, x2: Should have batch dimension
        :return: [ntk (ndarray), Jacobian (list(torch))]
        """
        params = {}
        for name, param in self.named_parameters():
            if f"blocks.{block_idx}" in name:
                params[name] = param.detach()

        # params = {k: v.detach() for k, v in self.named_parameters()}

        def fnet_single(params, x):
            return functional_call(self, params, (x.unsqueeze(0), int(block_idx), tau, softmax)).squeeze(0)

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
    """ REIN MtrNet Agent"""
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, learning_rate=1e-3,
                 dynamical_layer_param=False, lightMTR=True ):
        self.horizon = horizon
        self.policy = MtrNet(n_inputs, n_outputs, hidden_dim, horizon, dynamical_layer_param=dynamical_layer_param)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = self.policy.sigma_b
        self.lr = learning_rate
        self.horizon = horizon
        self.game_name = game_name
        self.name = "ReinMtrNet"
        self.lightMTR=lightMTR
        self.dynamical = dynamical_layer_param
    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.lr = lr

    def select_action(self, state, step):
        state_tensor = torch.tensor(state, dtype=torch.float)
        block_idx = self.horizon - step  # step starts from 1
        with torch.no_grad():
            action_probs = self.policy(state_tensor, block_idx, tau=self.tau)
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

                if not self.lightMTR:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[:block_idx+1].parameters():
                        param.requires_grad = True
                else:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[block_idx].parameters():
                        param.requires_grad = True

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor, block_idx, tau=self.tau)

                # Negative log likelihood of the taken action
                loss = -torch.log(action_probs[0][action]) * returns[- step]
                self.optimizer.zero_grad()
                loss.backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=8)

                self.optimizer.step()

                if not self.lightMTR:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[:block_idx+1].parameters():
                        param.requires_grad = False
                else:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[block_idx].parameters():
                        param.requires_grad = False

        return total_reward / len(episodes), total_reward / len(episodes)

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
    """MtrNet Agent"""
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, learning_rate=1e-3,
                 dynamical_layer_param=False, lightMTR=True):
        self.policy = MtrNet(n_inputs, n_outputs, hidden_dim, horizon, dynamical_layer_param=dynamical_layer_param)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = self.policy.sigma_b
        self.lr = learning_rate
        self.horizon = horizon
        self.game_name = game_name
        self.name = "MtrNet"
        self.lightMTR = lightMTR
        self.dynamical = dynamical_layer_param

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.SGD(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
            self.lr = lr

    def select_action(self, state, step):
        state_tensor = torch.tensor(state, dtype=torch.float)
        block_idx = self.horizon - step  # step starts from 1
        with torch.no_grad():
            action_probs = self.policy(state_tensor, block_idx, tau=self.tau)
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_probs.to("cpu"), [action_value]
        else:
            return action, action_probs.to("cpu"), action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        total_entropy_reward = 0

        self.optimizer.zero_grad()
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
            # entropy_rewards[-1] = entropy_rewards[-1] - self.policy.value(torch.FloatTensor(last_state),self.horizen - last_state ,self.tau).detach().numpy()
            C = np.cumsum(entropy_rewards[::-1])
            total_entropy_reward += C[-1]
            C = torch.FloatTensor(C)



            # Disable gradient computation for all parameters
            for param in self.policy.parameters():
                param.requires_grad = False

            for state, step, action, _, _ in episode:

                block_idx = self.horizon - step

                """if not self.lightMTR:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[:block_idx+1].parameters():
                        param.requires_grad = True"""

                # Enable gradient only for the block related to the current step
                for param in self.policy.blocks[block_idx].parameters():
                        param.requires_grad = True

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor, block_idx, tau=self.tau)

                # Compute the policy gradient loss
                loss = CustomLoss(C[block_idx])

                loss(action_probs[0][action]).backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)


                if self.lightMTR:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[block_idx].parameters():
                        param.requires_grad = False
                    """# Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[:block_idx+1].parameters():
                        param.requires_grad = False"""

        if not self.lightMTR:
                self.optimizer.step()

        return total_reward / len(episodes), total_entropy_reward/len(episodes)

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


class ShortLongAgent:
    """
    Short-Long Net Agent
    """
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, perc_list=None, tau=1, learning_rate=1e-3,
                 dynamical_layer_param=False, lightMTR=True):
        self.horizon = horizon
        if perc_list == None:
            self.perc_list = [0.03, 0.06, 0.1, 0.16, 0.26, 0.38, 0.5, 0.65, 0.8]
        else:
            self.perc_list = perc_list
        self.n_blocs = len(self.perc_list) + 1
        self.policy = MtrNet(n_inputs, n_outputs, hidden_dim, self.n_blocs, dynamical_layer_param=dynamical_layer_param)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.tau = tau
        self.beta = self.policy.sigma_b
        self.lr = learning_rate
        self.game_name = game_name
        self.name = "ShortLongNet"
        self.lightMTR = lightMTR
        self.dynamical = dynamical_layer_param

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.SGD(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
            self.lr = lr

    def select_action(self, state, step):
        state_tensor = torch.tensor(state, dtype=torch.float)
        block_idx = self.block_idx(self.horizon - step)  # step starts from 1
        with torch.no_grad():
            action_probs = self.policy(state_tensor, block_idx, tau=self.tau)
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_probs.to("cpu"), [action_value]
        else:
            return action, action_probs.to("cpu"), action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        total_entropy_reward = 0
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
            # entropy_rewards[-1] = entropy_rewards[-1] - self.policy.value(torch.FloatTensor(last_state),self.horizen - last_state ,self.tau).detach().numpy()
            C = np.cumsum(entropy_rewards[::-1])
            total_entropy_reward += C[-1]
            C = torch.FloatTensor(C)
            # Policy gradient update
            for state, step, action, _, _ in episode:
                block_idx = self.block_idx(self.horizon - step)

                if not self.lightMTR:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[:block_idx+1].parameters():
                        param.requires_grad = True
                else:
                    for param in self.policy.blocks[block_idx].parameters():
                        param.requires_grad = True

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor, block_idx, tau=self.tau)

                # Compute the policy gradient loss
                loss = CustomLoss(C[self.horizon - step])
                self.optimizer.zero_grad()
                loss(action_probs[0][action]).backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)

                self.optimizer.step()

                if not self.lightMTR:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[:block_idx+1].parameters():
                        param.requires_grad = False
                else:
                    # Enable gradient only for the block related to the current step
                    for param in self.policy.blocks[block_idx].parameters():
                        param.requires_grad = False

        return total_reward / len(episodes), total_entropy_reward/len(episodes)

    def block_idx(self, horizon_step):
        perc_hor_list = np.array(self.perc_list) * self.horizon
        i = 0
        for perc in perc_hor_list:
            if horizon_step <= perc_hor_list[i]:
                break
            else:
                i += 1
        return i

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
