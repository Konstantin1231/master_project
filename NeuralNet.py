import torch
import torch.nn as nn
import copy
import pickle
import torch.optim as optim
from enviroment import *
from ntk import empirical_ntk_ntk_vps, empirical_ntk_jacobian_contraction
from torch.func import functional_call


# Adding Cunstom loss, to ensure that optimizer works correctly
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


# Step 2: Define the policy model.
# A neural network which, given a state, outputs a probability distribution over actions.
class PolicyNet(nn.Module):
    def __init__(self, n_inputs = 1, n_outputs= 1, hidden_dim= 2):
        super(PolicyNet, self).__init__()
        # Defining a simple neural network with one hidden layer.
        self.sigma_w = None
        self.sigma_b = None
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, n_outputs),  # Output layer
        )
        self.output_size = n_outputs
        self.softmax = nn.Softmax(dim=-1)
        self.ntk_init()
    def forward(self, x, tau=1, softmax=True):
        x = self.fc(x)  # dividing by tau before
        # we apply softmax
        if softmax:
            return self.softmax(x / tau)
        else:
            return x

    def ntk(self, x1, x2, tau, mode="full", softmax=False, show_dim_jac=False):
        """
        Neural Tangent Kernel
        :param mode: either "trace" or "full"
        :param x1, x2: Should have batch dimension
        """
        params = {k: v.detach() for k, v in self.named_parameters()}

        def fnet_single(params, x):
            return functional_call(self, params, (x.unsqueeze(0), tau, softmax)).squeeze(0)

        result = empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, mode, show_dim_jac=show_dim_jac)
        return result

    def count_parameters_by_layers(self):
        parameters = {}
        for name, parameter in self.named_parameters():
            parameters[name] = parameter.numel()  # numel returns the total number of elements in the tensor
        return parameters

    def total_number_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            total_params += parameter.numel()  # numel returns the total number of elements in the tensor
        return total_params

    # Initialize weights of NN according to the scheme presented on the  page 102 EPFL_TH9825.pdf
    def ntk_init(self, sigma_w=np.sqrt(2), sigma_b=0):
        """
        NTK parametrization.
        :param sigma_w: sigma_w * W
        :param sigma_b: sigma_b * bias
        :return: None
        """
        self.sigma_b = sigma_b
        self.sigma_w = sigma_w

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

    # Save and Load model parameters
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

    def value(self, x, tau):
        return tau * torch.log((torch.exp(self.fc(x) / tau).sum() / self.output_size))

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



class ReinforceAgent:
    def __init__(self, n_inputs, n_outputs, hidden_dim, game_name, horizon=100000, learning_rate=1e-3, tau=1):
        self.policy = PolicyNet(n_inputs, n_outputs, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.beta = self.policy.sigma_b
        self.tau = tau
        self.horizon = horizon
        self.game_name = game_name
        self.lr = learning_rate
        self.name = "REIN"

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.lr = lr

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_probs, [action_value]
        else:
            return action, action_probs, action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        total_reward = 0
        for episode in episodes:
            returns = []  # List to store the returns for each step
            G = 0
            # Calculate the returns by iterating through the episode in reverse
            for _, _, _, reward, _ in reversed(episode):
                total_reward += reward
                G = reward + gamma * G
                returns.insert(0, G)
            # Convert episode data into tensors for PyTorch calculations
            states, _, actions, _, _ = zip(*episode)
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            returns_tensor = torch.FloatTensor(returns)

            # Compute the loss
            log_probs = torch.log(self.policy(states_tensor))
            picked_log_probs = log_probs[range(len(actions)), actions_tensor]
            loss = -torch.sum(picked_log_probs * returns_tensor)  # Policy gradient loss

            # Perform backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)
            self.optimizer.step()

        return total_reward / len(episodes)

    def ntk(self, x1, x2, mode="full", batch=False, softmax=False, show_dim_jac=False):
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
        return self.policy.ntk(x1, x2, tau=self.tau, mode=mode, softmax=softmax, show_dim_jac=show_dim_jac)


class MTRAgent:
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon, game_name, tau=1, learning_rate=1e-3):
        self.policy = PolicyNet(n_inputs + 1, n_outputs, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.horizon = horizon
        self.tau = tau
        self.beta = self.policy.sigma_b
        self.lr = learning_rate
        self.game_name = game_name
        self.name = "MTR"

    def set_optimazer(self, lr=None):
        if lr == None:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.lr = lr
    def select_action(self, input_vector):
        input_vector_tensor = torch.tensor(input_vector, dtype=torch.float)
        with torch.no_grad():
            probs = self.policy(input_vector_tensor, tau=self.tau)
        action = torch.multinomial(probs, num_samples=1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, probs, [action_value]
        else:
            return action, probs, action

    def train(self, episodes, gamma=0.99, clip_grad=False):
        # Initialize total_loss to store cumulative loss over all episodes
        # Iterate over each episode in episodes
        total_reward = 0
        for episode in episodes:
            # Initialize lists to store rewards and action probabilities for the episode
            reward_list = []
            prob_action_list = []

            # Calculate the returns and action probabilities by iterating through the episode in reverse
            # Corresponds to sampling actions and collecting rewards in Algorithm 1
            for _, _, _, reward, a in reversed(episode):
                total_reward += reward
                reward_list.append(reward)
                prob_action_list.append(a)
            # Compute entropy rewards and the cumulative sum, C
            # Corresponds to the MPG update step and C_i computation in Algorithm 1
            entropy_rewards = np.array(reward_list) - self.tau * np.log(
                self.policy.output_size * np.array(prob_action_list))
            # Centre rewards by subtraction of the value function
            # last_state, _, _, _, _ = episode[-1]
            # entropy_rewards[-1] = entropy_rewards[-1] - self.policy.value(torch.FloatTensor(last_state), self.tau).detach().numpy()
            C = np.cumsum(entropy_rewards)

            # Convert C to a float tensor for PyTorch calculations
            C = torch.FloatTensor(C)
            # Extract episode data and convert them to tensors for PyTorch calculations
            input_vector, step, actions, _, _ = zip(*episode)
            input_vector_tensor = torch.FloatTensor(input_vector)
            actions_tensor = torch.LongTensor(actions)

            # Compute the log probabilities of the actions using the policy
            # Select the  probabilities corresponding to the taken actions
            probs = self.policy(input_vector_tensor, tau=self.tau)
            picked_log_probs = probs[range(len(actions)), actions_tensor]
            picked_log_probs = torch.flip(picked_log_probs, dims=[0])
            # Compute the policy gradient loss
            loss = CustomLoss(C)
            self.optimizer.zero_grad()
            loss(picked_log_probs).backward()
            if clip_grad:
                torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)
            self.optimizer.step()

        # Return the average loss across all episodes
        return total_reward / len(episodes)

    def ntk(self, x1, x2, mode="full", batch=False, softmax=False, show_dim_jac=False):
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
        return self.policy.ntk(x1, x2, tau=self.tau, mode=mode, softmax=softmax, show_dim_jac=show_dim_jac)
