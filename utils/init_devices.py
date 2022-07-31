import torch


class CUDADevices:
    """
    A helper class for setting up primary and secondary CUDA devices.

    :param threshold (float) - an acceptance threshold for devices with higher available memory (default: ~2GB)
    """
    def __init__(self, threshold: float = 2e9) -> None:
        self.threshold = threshold
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count()

        self.device = ''  # Primary device

    def set_device(self, custom_device: str = None) -> None:
        """Updates the device attribute for either CUDA or CPU based on GPU availability.
        (Optional) when a custom device is provided, device is automatically set to it."""
        if custom_device is None:
            if self.cuda_available:
                self.__set_single_gpu()
            else:
                self.__set_cpu()
        else:
            self.device = custom_device

    def __set_single_gpu(self) -> None:
        """Sets CUDA to a single GPU with the highest available memory."""
        device_ids, memory_sizes = self.__get_available_devices()
        most_available_memory_idx = memory_sizes.index(max(memory_sizes))
        self.device = device_ids[most_available_memory_idx]
        print(f"CUDA available. Device set to GPU -> '{self.device}'.")

    def __set_cpu(self) -> None:
        """Sets device attribute to CPU when CUDA is unavailable."""
        self.device = 'cpu'
        print(f"CUDA unavailable. Device set to CPU -> '{self.device}'.")

    def __get_available_devices(self) -> tuple:
        """Gets the device IDs and their memory size for devices with available memory."""
        device_ids = []
        memory_size = []
        for num in range(self.device_count):
            available_memory = torch.cuda.mem_get_info(f"cuda:{num}")
            if available_memory[0] > self.threshold:
                device_ids.append(f"cuda:{num}")
                memory_size.append(available_memory[0])
        return device_ids, memory_size
