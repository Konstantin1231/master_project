import gym
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
            nn.Linear(hidden_dim, n_outputs),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.output_size = n_outputs


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
        print("Recalling weights")
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


# Step 4: Calculate returns for each episode and update the policy.
def train_policy(policy, optimizer, episodes, gamma=0.99, clip_grad= True ):
    total_reward = 0
    for episode in episodes:
        returns = []  # List to store the returns for each step
        G = 0
        r = 0 #for plotting reward
        # Calculate the returns by iterating through the episode in reverse
        for _, _, reward in reversed(episode):
            total_reward+= reward
            G = reward + gamma * G
            returns.insert(0, G)
        # Convert episode data into tensors for PyTorch calculations
        states, actions, _= zip(*episode)
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)

        # Compute the loss
        log_probs = torch.log(policy(states_tensor))
        picked_log_probs = log_probs[range(len(actions)), actions_tensor]
        loss = -torch.sum(picked_log_probs * returns_tensor)  # Policy gradient loss

        # Perform backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        if clip_grad == True:
            torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=6)
        optimizer.step()


    return total_reward / len(episodes)
