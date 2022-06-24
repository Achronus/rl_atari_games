import pytest

from agents.ppo import PPO
from core.env_details import EnvDetails
from core.parameters import EnvParameters, ModelParameters, PPOParameters
from models.actor_critic import ActorCritic
from utils.helper import to_tensor, normalize

import torch
import torch.nn as nn
import torch.optim as optim


@pytest.fixture
def env_details() -> EnvDetails:
    return EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', img_size=128, stack_size=4))


@pytest.fixture
def model_params(env_details: EnvDetails) -> ModelParameters:
    network = ActorCritic(input_shape=env_details.input_shape, n_actions=env_details.n_actions)
    return ModelParameters(
        network=network,
        optimizer=optim.Adam(params=network.parameters(), lr=1e-3, eps=1e-3),
        loss_metric=nn.MSELoss()
    )


@pytest.fixture
def ppo_params() -> PPOParameters:
    return PPOParameters(gamma=0.99, update_steps=1, loss_clip=0.1, rollout_size=1,
                         num_agents=2, num_mini_batches=1)


@pytest.fixture
def ppo(env_details, model_params, ppo_params) -> PPO:
    return PPO(env_details, model_params, ppo_params, seed=1)


@pytest.fixture
def device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def test_ppo_act_valid(ppo, device) -> None:
    state = normalize(to_tensor(ppo.envs.reset()))
    action_probs = ppo.network(state.to(device))[0]
    preds = ppo.act(action_probs)
    assert type(preds) == dict


def test_ppo_generate_rollouts_valid(ppo) -> None:
    try:
        ppo.generate_rollouts()
        assert True
    except (ValueError, TypeError):
        assert False


def test_ppo_compute_rtgs_and_advantages_valid(ppo) -> None:
    tensor = torch.zeros((ppo.buffer.size, ppo.buffer.num_agents))
    rtgs, advantages = ppo.compute_rtgs_and_advantages()
    rtgs_valid = rtgs.shape == tensor.shape
    advantages_valid = advantages.shape == tensor.shape
    assert all([rtgs_valid, advantages_valid])


def test_ppo_learn_valid(ppo) -> None:
    try:
        ppo.learn()
        assert True
    except (ValueError, TypeError):
        assert False


def test_ppo_clipped_value_loss_valid(ppo, device) -> None:
    # Get state values
    state = normalize(to_tensor(ppo.envs.reset())).to(device)
    state_values = ppo.network(state)[1]

    # Compute other metrics
    data_batch = ppo.buffer.sample_batch(['state_values'])
    rtgs, advantages = ppo.compute_rtgs_and_advantages()
    batch_returns, batch_state_values = rtgs, data_batch.state_values
    value_loss = ppo.clipped_value_loss(state_values.cpu(), batch_returns.cpu(), batch_state_values.cpu())
    assert type(value_loss.item()) == float


def test_ppo_clip_surrogate_valid(ppo, device) -> None:
    # Get predictions
    state = normalize(to_tensor(ppo.envs.reset())).to(device)
    action_probs, state_values = ppo.network(state)
    preds = ppo.act(action_probs)

    # Compute other metrics
    data_batch = ppo.buffer.sample_batch(['log_probs'])
    ratio = torch.exp(preds['log_prob'] - data_batch.log_probs)
    advantages = ppo.compute_rtgs_and_advantages()[1]
    policy_loss = ppo.clip_surrogate(ratio, advantages).mean()
    assert type(policy_loss.item()) == float


def test_ppo_add_to_buffer_valid(ppo) -> None:
    tensor = torch.ones((ppo.buffer.size, ppo.buffer.num_agents))
    ppo.add_to_buffer(0, rewards=tensor)
    assert all(torch.eq(ppo.buffer.rewards, tensor).tolist())


def test_ppo_train_invalid(ppo) -> None:
    try:
        ppo.train(num_episodes=1)
        assert False
    except AssertionError:
        assert True


def test_ppo_train_valid(ppo) -> None:
    try:
        ppo.train(num_episodes=2)
        assert True
    except (ValueError, RuntimeError):
        assert False
