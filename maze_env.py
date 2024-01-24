import gym
from gym import spaces
import numpy as np
import random

class CustomMazeEnv(gym.Env):
    """
    Maze environment
    """
    def __init__(self, maze_size=5, reward_pos=(4, 4), wall_pose = (2,2), reward=20, one_hot_coded=True):
        super(CustomMazeEnv, self).__init__()

        # Define the maze dimensions (e.g., 5x5 grid)
        self.n_actions = 4
        self.grid_size = maze_size
        self.reward_pose = reward_pos
        self.reward  = reward
        self.wall_pose = wall_pose
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Initialize agent's position
        self.agent_position = [0, 0]

        # Define the maze layout (0: empty, 1: obstacle)
        self.maze = np.zeros((self.grid_size, self.grid_size))
        self.maze[wall_pose[0], wall_pose[1]] = 1  # Example obstacle
        self.one_hot_coded = one_hot_coded

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
        if self.agent_position == self.reward_pose:
            reward = self.reward  # Agent reaches the goal
            done = True
        else:
            reward = - 1
            done = False
        if self.one_hot_coded:
            return np.eye(self.grid_size**2)[self.agent_position[1]*self.grid_size + self.agent_position[0], :], reward, done, {}, {}
        else:
            return list(self.agent_position), reward, done, {}, {}

    def reset(self):
        # Reset the agent's position to the starting point.
        # We do it random, between three starting points
        starting_points = [[0,0], [self.grid_size-1,0], [0,self.grid_size-1]]
        self.agent_position = random.choices(starting_points)[0]

        if self.one_hot_coded:
            return np.eye(self.grid_size**2)[self.agent_position[1]*self.grid_size + self.agent_position[0], :], {}
        else:
            return list(self.agent_position), {}

    def render(self):
        # Implement visualization of the maze (optional)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) == self.agent_position:
                    print("A", end=" ")  # Agent
                elif (row, col) == self.wall_pose:
                    print("G", end=" ")  # Goal
                elif self.maze[row, col] == 1:
                    print("#", end=" ")  # Wall
                else:
                    print(".", end=" ")  # Open space
            print()# Newline to separate rows
        print()


# Create an instance of the custom maze environment
env = CustomMazeEnv()
