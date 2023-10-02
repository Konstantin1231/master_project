import gym
from gym import spaces
import numpy as np


class CustomMazeEnv(gym.Env):
    def __init__(self):
        super(CustomMazeEnv, self).__init__()

        # Define the maze dimensions (e.g., 5x5 grid)
        self.grid_size = 5
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Initialize agent's position
        self.agent_position = (0, 0)

        # Define the maze layout (0: empty, 1: obstacle)
        self.maze = np.zeros((self.grid_size, self.grid_size))
        self.maze[2, 2] = 1  # Example obstacle

    def step(self, action):
        # Implement the dynamics of the maze
        # Update agent's position based on action
        if action == 0:  # Move up
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Move down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:  # Move left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:  # Move right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)

        # Check if the new position is valid (not outside the maze or blocked)
        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            if self.maze[new_position[0], new_position[1]] == 0:
                self.agent_position = new_position

        # Define a simple reward structure (e.g., reaching a goal)
        if self.agent_position == (4, 4):
            reward = 1.0  # Agent reaches the goal
            done = True
        else:
            reward = 0.0
            done = False

        return self.agent_position, reward, done, {}

    def reset(self):
        # Reset the agent's position to the starting point
        self.agent_position = (0, 0)
        return self.agent_position

    def render(self):
        # Implement visualization of the maze (optional)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) == self.agent_position:
                    print("A", end=" ")  # Agent
                elif (row, col) == (4, 4):
                    print("G", end=" ")  # Goal
                elif self.maze[row, col] == 1:
                    print("#", end=" ")  # Wall
                else:
                    print(".", end=" ")  # Open space
            print()  # Newline to separate rows



# Create an instance of the custom maze environment
env = CustomMazeEnv()
