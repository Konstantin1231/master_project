# Matryoshka Algo

**Note:** This project is an implementation of the Matryoshka and "Shared" Matryoshka algorithms with Neural Network policy approximators. As a part of the Master's thesis, it was design to implement the experimental phase in the work ["Neural Horizons: Exploring Matryoshka Policy Gradients"](./documents/Master_project_Medyanikov.pdf) (See section 3.3 Experiments)
 

## Table of Contents
- [Introduction](#introduction)
- [File Structure](#file-structure)
- [Download](#download)
- [Quick start](#quick-start)
- - [main.ipynb](#mainipynb)
- - [Environment settings](#change-environment-settings)
- - [Neural Network architecture](#change-neural-network-architecture)
- [Agents](#agents)
- [Environments](#environments)
- [CK/NTK analysis](#ckntk-analysis)
- [Update 17.10.2023](#update-17102023)
-- [Update 24-10.2023](#update-24102023)
-- [Update 31.10.2023](#update-31102023)

## Introduction

The Matryoshka Algo repository contains two implementations of the Matryoshka algorithm, each differing in the way we contract the approximation function. These implementations are:

1. **Full Connected Neural Network (NN):** In this implementation, we use a fully connected neural network with an extra dimension to encode the step variable. This approach provides quite naive but easy implementation.

2. **Special Case of ResNet NN, named MtrMet:** The second implementation leverages a special case of a Residual Neural Network (ResNet) architecture. This choice offers a different perspective on the algorithm's implementation. Which is closer, to the idea of the original algorithm or what we refer to as "Shred" Matryoshka design.

In addition to the Matryoshka algorithm implementations, we have also included an implementation of the REINFORCE algorithm for comparison purposes. As a result, you will find following classes (Agents) in this repository:

- `ReinforceAgent (REIN)`: This class represents the implementation of the REINFORCE algorithm.
- `MTRAgent (MTR)`: Similar to ReinforceAgent, with extra dimension for input, that used to encode the step horizon and entropy regularized reward.
- `OriginalMtrAgent (Original)`: Parametrization consist of a number of independed blocks, each representing NN with entropy regularization. Each block, encode policy of specific horizon step.
- `MtrNetAgent (MtrNet)`: Our novel design to realize Matroyshka features. (see more Section 3.1 "Shared NN" in ["Neural Horizons: Exploring Matryoshka Policy Gradients"](./documents/Master_project_Medyanikov.pdf))
 

## File Structure

| File/Folder Name     | Description                                                                                                                                                                                                                                                                           |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main.ipynb`         | Main script for initiating analysis.                                                                                                                                                                                                                                                  |
| `NeuralNet.py`       | Contains the neural network model class PolicyNet and two Agent's classes that use this NN: ReinforceAgent and MTRAgent.                                                                                                                                                              |
| `original.py`        | Contains the neural network model class OriginalMTR and  Agent's class that uses this NN: ReinforceAgent and OriginalMtrAgent.                                                                                                                                                        |
| `MtrNet.py`          | Contains the neural network model class MtrNet and Agent's classes that use this NN: ReinforceMtrNetAgent, MtrNetAgent and ShortLongAgent .                                                                                                                                           |
| `enviroment.py`      | Stores essential functionality to smoothly run different game's environments.                                                                                                                                                                                                         |
| `utils.py`           | Stores essential functionality to run training/testing loops. In addition, contains the Ntk_analysis class, that used to provide tools for spectral decomposition of NTK and CK kernels, estimate state distribution and estimate optimal projection on the effective rank of the CK. |
| `requirements.txt`   | Lists all the Python dependencies required for the project.                                                                                                                                                                                                                           |
| `/documents`         | Additional documentation and resources.                                                                                                                                                                                                                                               |
| `/images`            | Saved plots and figure, produced for the master's thesis.                                                                                                                                                                                                                             |
| `/results`           | Folder containing .csv files, recorder during experimental phase for master's thesis.                                                                                                                                                                                                 |
| `toy.py/maze.py`     | Custom made environments.                                                                                                                                                                                                                                                             |
| `ntk.py`             | Neural Target Kernel realizations. (Based on the work ["Fast Finite Width Neural Tangent Kernel"](./documents/fast_finite_width_NTK.pdf))                                                                                                                                             |
| `result_stats.ipynb` | Jupyter notebook to visualize results from the folder ./results.                                                                                                                                                                                                                      |
| `README.md`          | This file, containing an overview and guide to the repository.                                                                                                                                                                                                                        |

## Download

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
   
# Quick start
 
### main.ipynb
Open jupyter notebook main.ipynb.
Select the target environment and run cell. In addition, one can choose to set env.random = True to force the random initializing point.

<img src=images/img_1.png  width="400" >

Next, we set hyperparameters for agents, including learning rate, tau, number of episodes per epoch and number of epochs. By running this cell, we will initialize agents, in other words initialize the Neural Networks, that used to parametrize their policies.

<img src=images/img.png  width="400" >

Then, we configure the DataFrame to store results. One can use blank result.csv file for FROM_PATH, and crate new one for TO_PATH = <YOUR PATH>, where all obtained results will be dump in.

<img src=images/img_2.png  width="400" >

Then, one can continue by running the other cells containing training/testing loops for all agents. 
To visualize results, we use RENDER cell, where the only variable to set: agent - agent we want render and agent.tau = the temperature factor during the game.\
<img src=images/img_4.png  width="400" >


### Change environment settings
In order to make more changes, one should explor environment.py file -> game_setup() function.
Where, one can redefine any proposed constants or add new environment.

<img src=images/img_5.png  width="400" >

One can redefine default rewards using environment.py -> custom_reward() function.

<img src=images/img_6.png  width="400" >


### Change Neural Network architecture 
By default, each Neural Network block parametrized by Feed Forward Relu NN with two hidden layers. One, can modify it changing the .Q attribute in SimpleBlock class for each design (Original in original.py, MtrMet in MtrNet.py, REIN in NeuralNet.py).

<img src=images/img_7.png  width="400" >

Make sure, to keep consistency with self.output_layer.



# Agents

### Description of Agent Classes

Files: NeuralNet.py, original.py and MtrNet.py.

The Agent's classes in this project are designed to implement the following four algorithms:

1. **REIN (Class name: ReinforceAgent)**: Implements the standard REINFORCE algorithm with an \(\epsilon\)-greedy policy. This serves as a benchmark algorithm, representing stationary policies without the concept of step horizon.

2. **MTR (Class name: MTRAgent)**: Similar to REIN, MTR employs a singular policy but introduces an extra input dimension to incorporate current horizon step information into the policy network. This algorithm adapts to step horizon dynamics and can be viewed as an initial realization of the Matryoshka algorithm.

3. **Original (Class name: OriginalMtrAgent)**: A direct implementation of the Matryoshka principle, using a set of independent Neural Networks for each step horizon.

4. **MtrNet (Class name: MtrNetAgent)**: A novel implementation of the Matryoshka principle, utilizing shared parametrization with the S-MPG update.

All of these classes exhibit the same methods, which are outlined below.

### Agent class methods

| Method Name                 | Description |
| --------------------------- | ----------- |
| `__init__`                  | Initializes the agent with specified inputs, outputs, dimensions, horizon, game name, learning rate, and policy. |
| `set_optimizer`             | Sets or updates the learning rate for the optimizer. |
| `select_action`             | Selects an action based on the current state and step, using the policy. |
| `train`                     | Trains the agent over a specified number of episodes, updating the policy using gradient descent. |
| `ntk`                       | Computes the Neural Tangent Kernel (NTK) for given inputs and game steps. |

### Policy Network methods (Accessed through AgentClass.policy)

| Method Name                 | Description |
| --------------------------- | ----------- |
| `__init__`                  | Initializes the Original Neural Network Construction with specified inputs, outputs, hidden dimensions, and blocks. |
| `forward`                   | Forward pass for the neural network based on the input and horizon step. |
| `ntk_init`                  | Initializes weights of the neural network according to a specific scheme. |
| `value`                     | Computes the value function for a given input and horizon step. |
| `conjugate_kernel`          | Computes the conjugate kernel for specified inputs and a block index. |
| `ntk`                       | Computes the Neural Tangent Kernel (NTK) for given inputs, block index, and other parameters. |
| `count_parameters_in_block` | Counts the total number of parameters in a specified block of the neural network. |
| `total_number_parameters`   | Calculates the total number of parameters in the neural network. |
| `save_parameters`           | Saves the model parameters to a specified file path. |
| `load_parameters`           | Loads model parameters from a specified file path. |
| `norm_param`                | Prints the norm of parameters per layer in the neural network. |
| `rescale_weights`           | Rescales weights of the neural network. |
| `store_weights`             | Stores the current state of the neural network weights. |
| `compute_change_in_weights` | Computes the change in weights compared to a given old weight state. |
| `restore_weights`           | Restores weights of the neural network to a previously stored state. |

These methods provide a comprehensive set of functionalities for implementing and experimenting with the specified reinforcement learning algorithms in the context of policy gradient updates.

### MtrNet 
This class, have two additional attributes: 
- .LightMTR = (default) True. Enable Light MPG update instead of S-MPG one. (See more in Section 2.2 "Light MPG" in ["Neural Horizons: Exploring Matryoshka Policy Gradients"](./documents/Master_project_Medyanikov.pdf)).
- .dynamical = (default) False. Set decreasing order for the number of trainable, w.r.t step horizon. 


# Environments

To test Agent's performance we adapted four different environments (see below). Each environment offers unique challenges and characteristics that make them suitable for evaluating different RL algorithms.
**Note**: To set up hyperparameters for each environment, use ``game_setup`` function in enviroment.py. To set up rewards use function `enviroment.py>custom_reward` and to customize inputs, use ``enviroment.py>env_input_modification`` function.

### Cart-Pole Environment

The Cart-Pole environment is a classic benchmark in Reinforcement Learning, involving the task of balancing a pole on a moving cart.

**Description:**
- **State Space:** The state space consists of four continuous variables representing the position and velocity of the cart and pole.
- **Action Space:** The agent can take two discrete actions: push the cart left or right.
- **Rewards:** The standard reward function provides a reward of +1 for each time step the pole remains upright. If the pole falls or the cart moves too far from the center, the episode terminates with a reward of 0.

### Maze Environment

The Maze environment is a grid-world scenario where the goal is to navigate through a maze to reach a reward. It offers a discrete state and action space and is commonly used for testing RL algorithms in a discrete setting.

**Description:**
- **Grid Size:** The maze is a 5x5 grid with walls blocking certain paths.
- **Initial States:** The agent can start from one of three initial positions: (0,0), (0,4), or (4,0).
- **Goal:** The reward is positioned at (4,4) with a reward of +10. All other states provide a reward of -1.
- **Obstacle:** A wall is present at position (2,2), creating a barrier that the agent must navigate around.

### Lake Environment

The Lake environment is another grid-world scenario focusing on navigation and risk. The agent must avoid falling into holes while aiming to reach a rewarding destination. Additionally, the environment is slippery, introducing stochasticity into the agent's actions.

**Description:**
- **Grid Size:** The Lake is a 4x4 grid.
- **Initial State:** The agent starts at position (0,0).
- **Holes:** There are four holes at positions (1,1), (0,3), (3,1), and (3,2). Falling into a hole terminates the episode with a reward of -5.
- **Reward:** A high-reward state is located at (3,3) with a reward of +30. All other states provide a reward of -1 per time step.
- **Slippery:** The environment is slippery, meaning that actions may not always lead to the intended outcome, introducing randomness into the agent's movements.

### Toy Environment

This environment is designed with specific mechanics to test reinforcement learning algorithms. 

**Description:**
- **State Space:** The environment consists of a set of states `S = {0,1,2, ..., m-1}`, with a randomly selected initial state.
- **Action Space:** At each time step `t ∈ N`, the agent may choose between two actions `A = {+1, +2}`.
- **State Transitions:** The state transitions are defined as either `S(t+1) = (S(t) + 1) mod m ∈ S` or `S(t+1) = (S(t) + 2) mod m ∈ S`.
- **Rewards:** Correspondingly, for each pair `(a,s) ∈ A×S`, an immediate reward `r(a,s) ∈ R` is assigned.
- **Initial State:** The agent starts at a random position on the state space. For the **env.testing=True**, two initial states `S_0 = {0,2}` are chosen randomly with a 50% chance for each.
- **Goal** For the fixed horizon 'n', get the maximal cumulative reward.

This environment, with its simple yet intricate design, allows for a comprehensive evaluation of an RL agent's ability to learn optimal policies under various state and action scenarios.

## Update 17.10.2023

### TOY environment 

In recent update, we have added new game environment "Toy".
The object class can be found in the toy.py.
The environment has three variables:
- alphas; List of real numbers, used to construct rewards (Q_1): rewards = dot(alphas, basis ). In addition, the state dimension = len(alphas)
- random_basis; Boolean. When set to the True, will generate random orthonormal basis. By default, we use canonical basis e_1, ...
- one_hot_coded; Boolean. When set to the True, use one hot coded representation of integers. By default, we use integer representation of stats state = 0,1, ...len(alphas)-1

The environment setup could be done in the utils.py (game_setup() function):



# CK/NTK analysis
**Note**. Only for the Toy environment.

The `Ntk_analysis` class in `utils.py` provides tools for conducting analysis in the context of Neural Tangent Kernel (NTK) and Conjugate Kernel (CK). Key functionalities include:- eigen value/vector decomposition. (stored in attr: ``.eigen_vectors/.eigen_values``)
- estimate agent's state distribution (stored in attr: ``.m``)
- Kernel Rank (stored in attr: ``.rank``)
- Q-values for provided Agent's policy (stored in attr: ``.Q_pi``)
- optimal Q-values, as projection of Q_pi on the spectral decomposition of the Conjugate Kernel (stored in attr: ``.Q_opt``)
**Note**. Optimal Q-values, can be estimated only for Original and MtrMet/MtrNet Light agents.

Summary of class methods:

| Method                     | Description                                                                                                                 |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `ntk_matrix(Agent, idx_block)` | Calculates the NTK kernel matrix for a given block index of an agent. Returns a tensor of dimensions (n_inputs * n_actions, n_inputs * n_actions). |
| `ck_matrix(Agent, idx_block)` | Calculates the Conjugate Kernel matrix for a given block index of an agent. Similar in return dimensions to `ntk_matrix`.   |
| `ranks(mat)`               | Determines the matrix rank of a given matrix.                                                                               |
| `decompose(mat)`           | Performs eigenvalue/vector decomposition on the provided matrix.                                                            |
| `full_analysis(Agent)`     | Conducts a full analysis on an agent using either NTK or CK matrices. Populates attributes like eigenvalues, eigenvectors, and rank. |
| `estimate_m(Agent, n_runs)` | Estimates the state distribution of the agent over a specified number of runs.                                              |
| `generate_agent_policy_matrix(Agent)` | Generates a matrix representing the policy of an agent across different horizon steps.                                    |
| `generate_Q_pi(policy, Agent, step_hor)` | Generates Q-values for a provided agent's policy. Successive calls update Q-values for increasing horizon steps.         |
| `projection(Q, basis, pi, m)` | Computes the projection of Q-values on the spectral decomposition of the Conjugate Kernel.                                  |
| `genarate_optimal_Q(Agent, basis)` | Generates optimal Q-values, as the projection of `Q_pi` on the spectral decomposition of the Conjugate Kernel.           |
| `generate_optimal_policy_matrix(Agent, Q_opt)` | Generates an optimal policy matrix based on the optimal Q-values.                                                        |
| `policy_eval(pi_agent, pi_opt)` | Evaluates the policy by comparing the agent's policy matrix with the optimal policy matrix.                                 |

Usage example:
```sh 
ck = NTKanalysis(env, mode="ck")
ck.full_analysis(Agent)
ck.generate_optimal_Q(Agent)
````





![](images/picture1.png)

In addition, the Toy.render() method, will prepare a plot, that shows the current agent's position, and highlights the next optimal action.

![](images/picture2.png)



## Update 24.10.2023

### General:
 - one_hot_encoded option for Maze environment 
 - fixed crashes, related to the out of range index in early terminated episodes "Cart"/"Pendulum" 
 - MtrNet.forward() now has step parameter as input. And calculate only policies up to "step"
 - .forward() for all NN, now has boolean parameter "softmax", when set to False, forward will output the preference. When set to True, will apply softmax on the output. By default: True
 - MTRNet,now, has an option to set number of parameters per block in decreasing order. By setting "dynamical_layer_param=True" at agent initialization.

### NTK (Neural Tangent Kernel)
Each Agent has ability to call .ntk() method:

![](images/ntk.png)

By default, ntk do not accept batch inputs. So, one should set batch = True, if input tensors have batch dimensions. 
How it works:
   - horizon_step = horizon - step
   - find parameters theta_horizon_step (parameters used to produce policy for given horizon_step)
   - calculate jacobian (jac) of .forward(x_i, horizon_step, softmax=False) with respect to the theta_horizon_step
   - finally, ntk = jac(x1) @ jac^T(x2)

### Dynamical LR and TAU

In order to get rid-off repetitive code, I have added new "train_agent" function in utils.py. That will launch episodes run and training, and additionally will collect rewards gained after each epoch.

![](images/train_agent.png)

After "patience" number of epochs, we update learning rate and tau according to decay factor:

![](images/decay.png)

Remark: To get static tau and lr, set:
 - tau_end = agent.tau
 - lr_end = agent.lr

### ShortLongNet

Short-Long net is an adaptation of the MTRNet, to the environments with high horizon. 
Differently, from MTRNet, we do not associate a NN block with each horizon step. Instead, we associate one bloxk with a group of horizons steps.
At initialization of agent "ShortLongAgent", one should provide:
 *  perc_list = [0.03, 0.06, 0.1, 0.16, 0.26, 0.38, 0.5, 0.65, 0.8] #example
that indicates what is a percentile of horizon is used within different blocks. 
For example, for horizon_step that fall in the interval [0.06, 0.1]*horizon we use block #2. For [0.1, 0.16]*horizon, block #3.

## Update 31.10.2023

### New features for a Toy environment 
#### Q_star/V_star
Now one can generate and store Q_star and V_star values, using bew method: generate_all_q_stars(horizon, step_hor=2, tau=1)
Parameters:
- horizon: usually you want to set one, used during the training
- step-hor: Always starts with 2. (Needed for recurrence )
- tau: usually you want to set one, used during the training \
- output: dictionary .q_star and .v_star
Example

![](images/q_star.png)

#### New render function
- green circle: the optimal decision based on the optimal policy. Inside circle, we indicate the action value Q_star of the current step horizon.
- red circle: non-optimal choice. As well with the value of the current action value.
- arrow: the decision made by the agent. Green if the decision is the same with an optimal one.
- value inside current state, is a probability of the agent action choice. 

![](images/render_1.png)