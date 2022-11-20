from agents._agent import Agent
from agents.dqn import DQN
from agents.ppo import PPO
from agents.rainbow import RainbowDQN
from utils.helper import to_tensor, normalize


def video_render(agent: Agent, episodes: int = 5) -> None:
    """Watch a video representation of an agent in a given environment."""
    env = agent.env_details.make_env('testing', visualize=True)

    for episode in range(1, episodes+1):
        state, info = env.reset()
        done, truncated = False, False
        score = 0
        while not done or not truncated:
            state = normalize(to_tensor(state)).to(agent.device)
            if type(agent) == DQN or type(agent) == RainbowDQN:
                if type(agent) == RainbowDQN:
                    state = state.unsqueeze(0)
                state = agent.encode_state(state)
                action = agent.act(state)  # Generate an action
            elif type(agent) == PPO:
                state = agent.encode_state(state.unsqueeze(0))
                action_probs, _ = agent.network.forward(state)
                preds = agent.act(action_probs)  # Generate an action
                action = preds['action'].item()

            next_state, reward, done, truncated, info = env.step(action)  # Take an action
            state = next_state
            score += reward

        print(f'({episode}/{episodes}) Score: {int(score)}')
    env.close()
