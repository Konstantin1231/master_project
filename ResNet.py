import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.relu(x)
        x += residual # Residual connection

        return x

class ResNet(nn.Module):
    def __init__(self, input_dim, num_blocks, neurons_per_block):
        super(ResNet, self).__init__()
        self.blocks = nn.ModuleList([SimpleBlock(input_dim, neurons_per_block) for _ in range(num_blocks)])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        softmax_outputs = []
        for block in self.blocks:
            x = block(x)
            softmax_outputs.append(self.softmax(x))
        return softmax_outputs

class ReinforceResnetAgent:
    def __init__(self, n_inputs,  n_outputs, horizon,  hidden_dim, game_name, beta = 0.5, learning_rate=1e-3):
        self.horizon = horizon
        self.policy = ResNet(n_inputs, self.horizon, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ObsSpaceDim = n_inputs
        self.n_action = n_outputs
        self.beta = beta
        self.horizon = horizon
        self.game_name = game_name

    def select_action(self, state_tensor, step):
        softmax_outputs = self.policy(state_tensor)
        block_idx = self.horizon - step  # step starts from 1
        action_probs = softmax_outputs[block_idx]  # Using the softmax output of specific block
        action = torch.multinomial(action_probs, 1).item()
        if self.game_name == "Pendulum":
            possible_actions = np.linspace(-2, 2, self.n_action)
            action_value = possible_actions[action]
            return action, action_probs, [action_value]
        else:
            return action, action_probs, action

    def train(self, states, actions, rewards, episodes, gamma=0.99):
        total_reward = 0
        for episode in episodes:
            # Calculate the returns for each step in the trajectory
            returns = []
            G = 0

            for _, _, reward in reversed(episode):
                total_reward += reward
                G = reward + gamma * G
                returns.insert(0, G)

            returns = torch.FloatTensor(returns)

            # Policy gradient update
            self.optimizer.zero_grad()

            for idx, (state, action, _) in enumerate(zip(*episode)):
                # Zero the gradient for all parameters
                for param in self.policy.parameters():
                    param.grad = None

                # Enable gradient only for the block related to the current step
                block_idx = self.horizon - (idx + 1)
                for param in self.policy.blocks[block_idx].parameters():
                    param.requires_grad_(True)

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                softmax_outputs = self.policy(state_tensor)
                action_probs = softmax_outputs[block_idx]

                # Negative log likelihood of the taken action
                loss = -torch.log(action_probs[0][action]) * G
                loss.backward()

                # Disable gradient again for the block we just updated
                for param in self.policy.blocks[block_idx].parameters():
                    param.requires_grad_(False)

            self.optimizer.step()