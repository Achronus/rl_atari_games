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
        self.multi_devices = None

    def set_device(self, custom_device: str = None) -> None:
        """Updates the 'self.device' and 'self.multi_devices' attributes based on the number of CUDA devices available.
        (Optional) when a custom device is provided, 'self.device' is automatically set to it."""
        if custom_device is None:
            if self.cuda_available:
                if self.device_count > 1:
                    self.__set_multi_gpu()
                else:
                    self.__set_single_gpu()
            else:
                self.__set_cpu()
        else:
            self.device = custom_device

    def __set_multi_gpu(self) -> None:
        """Sets CUDA to multiple GPUs when more than one device is available."""
        device_ids = self.__get_multi_devices()
        self.device = device_ids[0]
        self.multi_devices = device_ids

    def __set_single_gpu(self) -> None:
        """Sets CUDA to a single GPU when available with 1 device."""
        self.device = "cuda:0"
        print(f"CUDA available. Device set to GPU -> '{self.device}'")

    def __set_cpu(self) -> None:
        """Sets device attribute to CPU when CUDA is unavailable."""
        self.device = 'cpu'
        print(f"CUDA unavailable. Device set to CPU -> '{self.device}'.")

    def __get_multi_devices(self) -> list:
        """
        Gets a list of available CUDA devices that are above the available memory threshold.
        Returns the ids as a list.
        """
        device_ids = []
        for num in range(self.device_count):
            available_memory = torch.cuda.mem_get_info(f"cuda:{num}")[0]
            if available_memory > self.threshold:
                device_ids.append(f"cuda:{num}")
        return device_ids
