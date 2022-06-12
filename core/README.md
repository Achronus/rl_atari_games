# Core Folder

This folder houses the core functionality of the application. 

One of the main files of note is the `create.py` file. The file provides an abstraction of functionality through the `create_model()` function that allows users to create DQN or PPO models efficiently. 
The function requires environment variables to be stored in a `.env` file, stored in the applications root directory.

This readme document intends to clarify the variables required within the `.env` file. We divide the readme into four categories of parameters: environment, generic, DQN and PPO.

## Environment

- `ENV_1` - a `string` containing the environment OpenAI Gym name. For example, `ALE/SpaceInvaders-v5`.
- `IMG_SIZE` - an `integer` specifying the new image dimension size after reshaping the state space. For example,
`128` converts states into `(128, 128)`.
- `STACK_SIZE` - an `integer` denoting the number of frames to stack together. In the Human-level control through deep reinforcement learning paper, the stack size is `4`. Stacking frames allows the agent to perceive movement.
- `CAPTURE_VIDEO` - a `boolean` value (`True` or `False`) denoting whether to record video snippets of the agent.

## Generic

- `GAMMA` - a `float` value representing the return discount factor.
- `LEARNING_RATE` - a `float` value for the PyTorch optimizer.
- `EPSILON` - a `float` value for the PyTorch optimizer.
- `SEED` - an `integer` value used for reproducing results.
- `UPDATE_STEPS` - an `integer` value denoting how often to update the network.

## DQN

- `TAU` - a `float` value used for soft updating the target network.
- `BUFFER_SIZE` - an `integer` value denoting the maximum replay buffer size.
- `BATCH_SIZE` - an `integer` value highlighting the number of batches to take from the buffer during training.
- `EPS_START` - a `float` value representing the initial epsilon value when taking greedy actions. We recommend setting this to `1.0`.
- `EPS_END` - a `float` value that denotes the final epsilon value. Acts as a threshold for always taking greedy actions.
- `EPS_DECAY` - a `float` value that indicates the epsilon decay rate each timestep.
- `MAX_TIMESTEPS` - an `integer` value representing the maximum number of timesteps taken in an episode.

  _(note)_ Official paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

## PPO

- `LOSS_CLIP` - a `float` value used for the surrogate clipping.
- `ROLLOUT_SIZE` - an `integer` value indicating the number of samples to train the agent on during each rollout step. 
- `NUM_AGENTS` - an `integer` value representing the number of agents/environments used during training. Used for parallelization.
- `NUM_MINI_BATCHES` - an `integer` value highlighting the number of mini-batches during training. Used to divide the batches into smaller chunks for more optimal learning.
- `ENTROPY_COEF` - a `float` value denoting the regularisation coefficient for entropy (lambda). `c1` in the official paper.
- `VALUE_LOSS_COEF` - a `float` value that acts as the value loss coefficient. `c2` in the official paper.
- `CLIP_GRAD` - a `float` value indicating the maximum value obtainable during gradient clipping.

  _(note)_ Official paper [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)