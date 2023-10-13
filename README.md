# Matryoshka Algo

**Note:** This project is a draft implementation of the Matryoshka algorithm, as described in the paper https://arxiv.org/pdf/2303.12785.pdf. Please be aware that this is a work in progress and may not be the final implementation.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Results](#results)
- [To-Do List](#to-do-list)

## Introduction

The Matryoshka Algo repository contains two implementations of the Matryoshka algorithm, each differing in the way we contract the approximation function. These implementations are:

1. **Full Connected Neural Network (NN):** In this implementation, we use a fully connected neural network with an extra dimension to encode the step variable. This approach provides quite naive but easy implementation.

2. **Special Case of ResNet NN:** The second implementation leverages a special case of a Residual Neural Network (ResNet) architecture. This choice offers a different perspective on the algorithm's implementation. Which is closer, to the idea of the original algorithm.

In addition to the Matryoshka algorithm implementations, we have also included an implementation of the REINFORCE algorithm for comparison purposes. As a result, you will find two main classes in this repository:

- `ReinforceAgent`: This class represents the implementation of the REINFORCE algorithm.
- `MTRAgent`: This class represents the Matryoshka algorithm implementations, allowing you to explore and compare the two different approaches.








## Setup

To set up this repository on your local machine, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Konstantin1231/master_project.git
   
2. Navigate to your Project and Create Virtual environment:
   ```sh
   cd Matryoshka-Algo
   python -m venv venv
3. Activate Virtual environment and install all dependencies:
   ```sh
   venv\Scripts\activate
   pip install -r requirements.txt

## Results

### Cart-Pole Game
The Cart-Pole game is a classic reinforcement learning environment where the goal is to balance a pole on a moving cart. The agent must make decisions to keep the pole balanced for as long as possible. The result shows the performance of our Matryoshka algorithm on the Cart-Pole task.

![](images/Cart_1.jpg)
![](images/Cart_4.jpg)

We see that ntk parametrization, that have Beta scalar variable to control chaos order has a significant impact on the training.

### Frozen Lake Game
Frozen Lake is another reinforcement learning environment where the agent navigates a grid world to reach a goal while avoiding holes in the ice. The result showcases how our algorithm performs in this challenging environment.

![](images/Lake.jpg)
![](images/Lake_1.jpg)

### Maze Game
The Maze game involves navigating through a complex maze to reach the goal. It's a test of the algorithm's ability to handle more intricate environments. This result demonstrates our algorithm's performance in solving mazes.

![](images/Maze.jpg)

## To-Do List

Here are the upcoming features and improvements planned for the Matryoshka Algo project:

1. **Working NTK for Fully Connected NN:** We aim to develop a functional Neural Tangent Kernel (NTK) implementation for the fully connected neural network used in our first Matryoshka algorithm implementation. 

2. **New Toy Environment:** We plan to create a new custom toy environment that aligns with the theoretical findings of the Matryoshka algorithm. This environment will serve as a dedicated testbed for further experimentation and validation.

3. **New Custom ResNet:** As mentioned in the introduction.











