import gym
import numpy as np
import torch
import torch.nn as nn
import copy
import pickle

# Step 1: Set up the environment.
env = gym.make('CartPole-v1')


# Step 2: Define the policy model.
# A neural network which, given a state, outputs a probability distribution over actions.
class PolicyNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim):
        super(PolicyNet, self).__init__()
        # Defining a simple neural network with one hidden layer.
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, hidden_dim),  # hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, n_outputs),  # Output layer
        )
        self.output_size = n_outputs

    def forward(self, x, tau=1):
        logits = nn.Sequential(nn.Softmax())(self.fc(x) / tau)  # dividing by tau before
        # we apply softmax
        return logits

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


def train_policy_mtr(policy, optimizer, episodes, gamma=0.99, tau=1):
    # Initialize total_loss to store cumulative loss over all episodes
    total_loss = 0

    # Iterate over each episode in episodes
    for episode in episodes:
        # Initialize lists to store rewards and action probabilities for the episode
        reward_list = []
        prob_action_list = []

        # Calculate the returns and action probabilities by iterating through the episode in reverse
        # Corresponds to sampling actions and collecting rewards in Algorithm 1
        for _, _, reward, a in reversed(episode):
            reward_list.append(reward)
            prob_action_list.append(a)

        # Compute entropy rewards and the cumulative sum, C
        # Corresponds to the MPG update step and C_i computation in Algorithm 1
        entropy_rewards = np.array([reward_list]) - tau * np.log(policy.output_size * np.array(prob_action_list))
        C = np.cumsum(entropy_rewards)

        # Convert C to a float tensor for PyTorch calculations
        C = torch.FloatTensor(entropy_rewards)

        # Extract episode data and convert them to tensors for PyTorch calculations
        input_vector, actions, _, _ = zip(*episode)
        input_vector_tensor = torch.FloatTensor((input_vector))
        actions_tensor = torch.LongTensor(actions)

        # Compute the log probabilities of the actions using the policy
        # Select the log probabilities corresponding to the taken actions
        log_probs = torch.log(policy(input_vector_tensor))
        picked_log_probs = log_probs[range(len(actions)), actions_tensor]

        # Compute the policy gradient loss
        loss = torch.sum(torch.flip(picked_log_probs, dims=[0]) * C)

        # Perform backpropagation and optimization step
        # Corresponds to the update of policy parameters in Algorithm 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item()

    # Return the average loss across all episodes
    return total_loss / len(episodes)

