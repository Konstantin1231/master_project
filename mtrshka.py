import gym
import numpy as np
import torch
import torch.nn as nn
import copy
import pickle
import torch.optim as optim
from enviroment import *
# Step 1: Set up the environment.
env = gym.make('CartPole-v1')

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
        return -torch.sum(torch.log(outputs)*self.C)


# Step 2: Define the policy model.
# A neural network which, given a state, outputs a probability distribution over actions.
class PolicyNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim):
        super(PolicyNet, self).__init__()
        # Defining a simple neural network with one hidden layer.
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, n_outputs),  # Output layer
        )
        self.output_size = n_outputs
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, tau=1):
        x = self.fc(x) # dividing by tau before
        # we apply softmax
        return self.softmax(x/tau)

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
        return tau * torch.log((torch.exp(self.fc(x) / tau).sum() / (self.output_size)))

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
    def __init__(self, n_inputs, n_outputs, hidden_dim, game_name, horizon = 100000, beta = 0.5, learning_rate=1e-3):
        self.policy = PolicyNet(n_inputs, n_outputs, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.beta = beta
        self.horizon = horizon
        self.game_name = game_name
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        action_probs = self.policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2,2,self.n_action)
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
            for _, _,_, reward, _ in reversed(episode):
                total_reward += reward
                G = reward + gamma * G
                returns.insert(0, G)
            # Convert episode data into tensors for PyTorch calculations
            states, _, actions, _,_ = zip(*episode)
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
            if clip_grad == True:
                torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)
            self.optimizer.step()

        return total_reward / len(episodes)


class MTRAgent:
    def __init__(self, n_inputs, n_outputs, hidden_dim, horizon,game_name, tau=1,beta = 0.5, learning_rate=1e-3):
        self.policy = PolicyNet(n_inputs+1, n_outputs, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.horizon =  horizon
        self.tau = tau
        self.beta = beta
        self.game_name = game_name
    def select_action(self, input_vector):
        input_vector_tensor = torch.tensor(input_vector, dtype=torch.float)
        with torch.no_grad():
            probs = self.policy(input_vector_tensor, tau = self.tau)
        action = torch.multinomial(probs, num_samples=1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2,2,self.n_action)
            action_value = possible_actions[action]
            return action, probs, [action_value]
        else:
            return action, probs, action

    def train(self, episodes, gamma=0.99,  clip_grad=False):
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
            entropy_rewards = np.array(reward_list) - self.tau * np.log(self.policy.output_size * np.array(prob_action_list))
            C = np.cumsum(entropy_rewards)

            # Convert C to a float tensor for PyTorch calculations
            C = torch.FloatTensor(C)
            # Extract episode data and convert them to tensors for PyTorch calculations
            input_vector, step, actions, _, _ = zip(*episode)
            input_vector_tensor = torch.FloatTensor((input_vector))
            actions_tensor = torch.LongTensor(actions)

            # Compute the log probabilities of the actions using the policy
            # Select the  probabilities corresponding to the taken actions
            probs = self.policy(input_vector_tensor)
            picked_log_probs = probs[range(len(actions)), actions_tensor]
            picked_log_probs = torch.flip(picked_log_probs, dims=[0])
            # Compute the policy gradient loss
            loss = CustomLoss(C)
            self.optimizer.zero_grad()
            loss(picked_log_probs).backward()
            if clip_grad == True:
                torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=10)
            self.optimizer.step()
            """
            for i in range(len(C)):
                loss = CustomLoss(C[i])
                # Perform backpropagation and optimization step
                # Corresponds to the update of policy parameters in Algorithm 1
                optimizer.zero_grad()
                print(C[i])
                loss(picked_log_probs[i]).backward()
                if clip_grad == True:
                    torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=10)
                optimizer.step()
                # Accumulate the loss
            """

        # Return the average loss across all episodes
        return total_reward / len(episodes)




