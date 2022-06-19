# Environment Variables

One of the main files of the application is the `create.py` file. The file provides an abstraction of functionality through the `create_model()` function, allowing users to create DQN or PPO models efficiently. 
The function requires environment variables to be stored in a `.env` file, housed in the applications root directory.

This README intends to clarify the variables required within the `.env` file.

## Environment

- `ENV_1` - a `string` containing the environment OpenAI Gym name. For example, `ALE/SpaceInvaders-v5`.
- `IMG_SIZE` - an `integer` specifying the new image dimension size after reshaping the state space. For example,
`128` converts states into `(128, 128)`.
- `STACK_SIZE` - an `integer` denoting the number of frames to stack together. In the Human-level control through deep reinforcement learning paper, the stack size is `4`. 
Stacking frames allows the agent to perceive movement. With an `IMG_SIZE` of 128 and `STACK_SIZE` of 4, states are defined as `(4, 128, 128)`.
- `CAPTURE_VIDEO` - a `boolean` value (`True` or `False`) denoting whether to record video snippets of the agent.
- `RECORD_EVERY` - an `integer` denoting the number of episodes between each stored video. Only required if `CAPTURE_VIDEO` is `True`. 
Can be used in conjunction with the `set_save_every()` function, in the `create.py` file, to dynamically set a `SAVE_EVERY` hyperparameter for the `model.train()` method.

## Generic

- `GAMMA` - a `float` value representing the return discount factor.
- `LEARNING_RATE` - a `float` value for the PyTorch optimizer.
- `EPSILON` - a `float` value for the PyTorch optimizer.
- `SEED` - an `integer` value used for reproducing results.
- `UPDATE_STEPS` - an `integer` value denoting how often to update the network.
- `CLIP_GRAD` - a `float` value indicating the maximum value obtainable during gradient clipping.

## DQN Core

- `TAU` - a `float` value used for soft updating the target network.
- `BUFFER_SIZE` - an `integer` value denoting the maximum replay buffer size.
- `BATCH_SIZE` - an `integer` value highlighting the number of batches to take from the buffer during training.

## DQN

- `EPS_START` - a `float` value representing the initial epsilon value when taking greedy actions. We recommend setting this to `1.0`.
- `EPS_END` - a `float` value that denotes the final epsilon value. Acts as a threshold for always taking greedy actions.
- `EPS_DECAY` - a `float` value that indicates the epsilon decay rate each timestep.

  _(Note)_ Official paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

## Rainbow DQN

- `N_STEPS` - an `integer` denoting the number of steps for multi-step (N-step) learning.
- `REPLAY_PERIOD` - an `integer` representing the number of transitions before learning starts.
- `LEARN_FREQUENCY` - an `integer` for the number of timesteps to perform agent learning. For example, a learning frequency of 4 results in the agent learning every 4 timesteps.
- `REWARD_CLIP` - a `float` value denoting the bounds for rewards. For example, 0.1 would clip rewards in the range of [-0.1, +0.1].
- `PRIORITY_EXPONENT` - a `float` value denoting the prioritized replay buffer exponent (alpha).
- `PRIORITY_WEIGHT` - a `float` stating the initial prioritized replay buffer importance sampling weight (beta).
- `N_ATOMS` - an `integer` indicating the number of atoms (distributions) in the Categorical DQN. Atoms are referred to as the 'canonical returns' of the value distribution.
- `V_MIN`, `V_MAX` - `integers` for the minimum and maximum size of the atoms. For example, [-10, +10].

_(Note)_ Official paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

## PPO

- `LOSS_CLIP` - a `float` value used for the surrogate clipping.
- `ROLLOUT_SIZE` - an `integer` value indicating the number of samples to train the agent on during each rollout step. 
- `NUM_AGENTS` - an `integer` value representing the number of agents/environments used during training. Used for parallelization.
- `ENTROPY_COEF` - a `float` value denoting the regularisation coefficient for entropy (lambda). `c1` in the official paper.
- `VALUE_LOSS_COEF` - a `float` value that acts as the value loss coefficient. `c2` in the official paper.
- `NUM_MINI_BATCHES` - an `integer` value highlighting the number of mini-batches during training. Used to divide the batches into smaller chunks for more optimal learning.

_(Note)_ Official paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)