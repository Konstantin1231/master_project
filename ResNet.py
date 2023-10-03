import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple ResNet block with only ReLU non-linearity and skip connections
class ResNetBlock(nn.Module):
    def __init__(self, in_features):
        super(ResNetBlock, self).__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.relu(out)
        out += residual
        return out


# Define the full ResNet model composed only of fully connected layers
class ResNet(nn.Module):
    def __init__(self, in_features, num_actions, hidden_layer, num_blocks):
        super(ResNet, self).__init__()
        self.fc_input = nn.Linear(in_features, hidden_layer)
        self.relu = nn.ReLU()
        self.resnet_blocks = nn.Sequential(*[ResNetBlock(hidden_layer) for _ in range(num_blocks)]) # We can conroll number of blocks
        self.fc_output = nn.Linear(hidden_layer, num_actions)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, tau = 1):
        out = self.fc_input(x)
        out = self.relu(out)
        out = self.resnet_blocks(out)
        out = self.fc_output(out)
        return out