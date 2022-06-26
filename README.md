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
|   +-- intrinsic.py
|   +-- ppo.py
|   +-- rainbow.py
+-- core
|   +-- buffer.py
|   +-- create.py
|   +-- env_details.py
|   +-- exceptions.py
|   +-- parameters.py
|   +-- README.md
|   +-- template.yaml
+-- models
|   +-- _base.py
|   +-- actor_critic.py
|   +-- cnn.py
|   +-- dueling.py
|   +-- linear.py
+-- tests
|   +-- test_buffer.py
|   +-- test_create.py
|   +-- test_dqn.py
|   +-- test_env_details.py
|   +-- test_helper.py
|   +-- test_logger.py
|   +-- test_models.py
|   +-- test_ppo.py
+-- utils
|   +-- dataloader.py
|   +-- helper.py
|   +-- init_devices.py
|   +-- logger.py
|   +-- model_utils.py
|   +-- plotter.py
|   +-- render.py
+-- .coverage
+-- .coveragerc
+-- LICENSE
+-- main.py
+-- parameters.yaml
+-- README.md
+-- requirements.txt
+-- rl_with_atari.ipynb
```

- `\agents` - contains the RL algorithm implementations
- `\core` - core functionality of the artefact
- `\models` - neural network models for the agents
- `\tests` - unit tests for the application
- `\utils` - utility classes and functions that provide extra functionality

## Dependencies

This project requires a Python 3.10 environment, which can be created with the following instructions:

1. Create (and activate) a new environment.

   - Linux or Mac

    ```bash
    conda create --name rlatari python=3.10
    source activate rlatari
    ```

   - Windows

   ```bash
   conda create --name rlatari python=3.10
   activate rlatari
   ```

2. Clone the repository, navigate to the `rl_atari_games/` folder and install the required dependencies.

    _(Note)_ a requirements.txt file is accessible within this folder detailing a list of the required dependencies.

    ```bash
    git clone https://github.com/Achronus/rl_atari_games.git
    cd rl_atari_games
    conda install -c conda-forge jupyterlab
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
    pip install gym[atari,accept-rom-license,other]
    pip install -r requirements.txt
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `rlatari` environment.

    ```bash
    python -m ipykernel install --user --name rlatari --display-name "rlatari"
    ```

4. Run the `jupyter-lab` command to start JupyterLab and access the Jupyter Notebook named `rl_with_atari.ipynb`, or run the `main.py` file.

    _(Note)_ running the `main.py` file will train the models. It is advised to examine this file before running it.


## References

The algorithms created in this repository have been inspired and adapted from the following developers:

Rainbow DQN -
- [Curt Park: Rainbow is all you need (GitHub)](https://github.com/Curt-Park/rainbow-is-all-you-need)
- [Kaixhin: Rainbow (GitHub)](https://github.com/Kaixhin/Rainbow/tree/1745b184c3dfc03d4ffa3ce2342ced9996b39a60)

PPO - 
- [ericyangyu: PPO for beginners (GitHub)](https://github.com/ericyangyu/PPO-for-Beginners)
- [vwxyzjn: PPO implementation details (GitHub)](https://github.com/vwxyzjn/ppo-implementation-details)

A special thanks to each of these developers for creating incredible work in the RL space. Please check out their original work using the links above.
