import os
import tarfile
import re

from core.env_details import EnvDetails
from core.parameters import AgentParameters
from utils.helper import to_tensor, number_to_num_letter, save_model
from utils.logger import Logger


class Agent:
    """A base class for all agents."""
    def __init__(self, env_details: EnvDetails, params: AgentParameters, device: str,
                 seed: int, logger: Logger) -> None:
        self.env_details = env_details
        self.params = params
        self.device = device
        self.seed = seed
        self.logger = logger

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
            filename (str) - a custom filename. Note: environment name and episode number are post-appended
            extra_data (dict) - additional items to store (e.g. network.state_dict())
        """
        if i_episode % save_count == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)
            env_name = self.save_file_env_name()  # Reduce environment name if needed

            filename += f'_{env_name}_ep{int(ep_idx)}{ep_letter.lower()}'
            # Create initial param_dict
            param_dict = dict(
                env_details=self.env_details,
                params=self.params,
                seed=self.seed
            )
            param_dict.update(extra_data)  # Update with extra info
            save_model(filename, param_dict)  # Save model
            print(f"Saved model at episode {i_episode} as: '{filename}.pt'.")

            self.__save_logger(filename, env_name)

    def __save_logger(self, filename: str, env_name: str) -> None:
        """Stores the agents logger object with its values to a compressed file."""
        name = f"{filename.split('_')[0]}_{env_name}_logger_data"  # Gets model name from filename
        storage_in = f"saved_models/{name}.pt"
        storage_out = f"saved_models/{name}.tar.gz"

        # Store logger to separate file
        save_model(name, dict(logger=self.logger))

        # Remove file if already exists
        if os.path.exists(storage_out):
            os.remove(storage_out)

        # Create new version of file
        with tarfile.open(storage_out, "w:gz") as tar:
            tar.add(storage_in, arcname=f"{name}.pt")

        os.remove(storage_in)  # Remove uncompressed file
        print(f"Saved logger data to '{storage_out}'. Total size: {os.stat(storage_out).st_size} bytes")

    @staticmethod
    def _count_actions(actions: list) -> dict:
        """Helper function that concatenates a list of Counter objects containing the number of actions
        taken per episode. Returns a single counter object containing the accumulated values."""
        for idx in range(1, len(actions)):
            actions[0] += actions[idx]
        return actions[0]

    @staticmethod
    def _calc_mean(data_list: list) -> float:
        """Helper function for computing the mean of a list of data. Returns the mean value."""
        return to_tensor(data_list).detach().mean().item()

    def log_data(self, **kwargs) -> None:
        """Adds data to the logger."""
        self.logger.add(**kwargs)

    def save_file_env_name(self, threshold: int = 6, num_chars: int = 3) -> str:
        """
        Reduces the environment name if it is larger than threshold. Returns the updated name.

        :param threshold (int) - value for comparing length of environment name, larger than this value reduces it
        :param num_chars (int) - number of letters for each word during reduction
        """
        # Reduce env name if large
        env_name = self.env_details.name
        if len(env_name) > threshold:
            env_name = re.findall('[A-Z][^A-Z]*', env_name)  # Uppercase letter split
            env_name = ''.join([item[:num_chars] for item in env_name])  # First num_chars of each word
        return env_name
