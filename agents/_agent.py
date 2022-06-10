from core.env_details import EnvDetails
from core.parameters import AgentParameters
from utils.helper import set_device, number_to_num_letter, save_model
from utils.logger import Logger


class Agent:
    """A base class for all agents."""
    def __init__(self, env_details: EnvDetails, params: AgentParameters,
                 seed: int, logger: Logger) -> None:
        self.env_details = env_details
        self.params = params
        self.seed = seed
        self.logger = logger

        self.device: str = set_device()

    def _initial_output(self, num_episodes: int, extra_info: str = '') -> None:
        """Provides basic information about the algorithm to the console."""
        assert isinstance(extra_info, str), "'extra_info' must be a string!"
        ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
        print(f'Training agent on {self.env_details.name} with '
              f'{int(ep_total_idx)}{ep_total_letter} episodes.')
        print(f'{extra_info}')

    def _save_model_condition(self, i_episode: int, save_count: int, filename: str, extra_data: dict) -> None:
        """
        Saves the model when the current episode equals the save count.

        Parameters:
            i_episode (int) - current episode number
            save_count (int) - episode number to save
            filename (str) - a custom filename. Note: episode number is post-appended
            extra_data (dict) - additional items to store (e.g. network.state_dict())
        """
        if i_episode % save_count == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)
            filename += f'_ep{int(ep_idx)}{ep_letter}'.lower()
            # Create initial param_dict
            param_dict = dict(
                env_details=self.env_details,
                params=self.params,
                logger=self.logger,
                seed=self.seed
            )
            param_dict.update(extra_data)  # Update with extra info
            save_model(filename, param_dict)  # Save model
            print(f"Saved model at episode {i_episode} as: '{filename}.pt'.")

    def log_data(self, **kwargs) -> None:
        """Adds data to the logger."""
        self.logger.add(**kwargs)
