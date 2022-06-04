# Reinforcement Learning with Atari Games
This repository focuses on exploring how Intrinsic Motivation (IM) effects Reinforcement Learning (RL) policy-based models. 
We use a Rainbow Deep Q-Network (RDQN) to simulate off-policy and Proximal Policy Optimization (PPO) as on-policy. 

We focus on three IM methods: curiosity, empowerment and surprise-based motivation, and evaluate their performance on three Atari games: 
Space Invaders, Q*bert, and Montezuma's Revenge. Our research aims to provide intuition on how IM methods affect these popular on-policy and off-policy agents.

![Atari Games](/imgs/atari-games.png)

_Figure 1. Examples of each environment. From left to right: Space Invaders, Q*bert, and Montezuma's Revenge (Source: [Gym Docs](https://www.gymlibrary.ml/environments/atari/))._

## File Structure
The file structure for the artefact is outlined below.

``` ANSI
+-- agents
|   +-- _agent.py
|   +-- dqn.py
+-- core
|   +-- buffer.py
|   +-- env_details.py
|   +-- parameters.py
|   +-- wrappers.py
+-- models
|   +-- _base.py
|   +-- cnn.py
+-- tests
|   +-- test_buffer.py
|   +-- test_dqn.py
|   +-- test_env_details.py
|   +-- test_helper.py
|   +-- test_models.py
+-- utils
|   +-- helper.py
|   +-- logger.py
|   +-- plotter.py
|   +-- render.py
+-- .coverage
+-- .coveragerc
+-- main.py
+-- rl_with_atari.ipynb
+-- LICENSE
+-- README.md
+-- requirements.txt
```
- `\agents`: contains the RL algorithm implementations
- `\core`: core functionality of the artefact
- `\models`: neural network models for the agents
- `\tests`: unit tests for the application
- `\utils`: utility classes and functions that provide extra functionality
