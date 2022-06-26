

class IntrinsicMotivation:
    """A class dedicated to Intrinsic Motivation reward modules. These modules act as extensions
    applicable to any RL model."""
    def __init__(self, im_type: str) -> None:
        self.type = im_type
        self.valid_types = ['curiosity', 'empowerment', 'surprise-based']
        self.reward_method = self.__reward_controller()

    def __reward_controller(self) -> callable:
        """Controller for applying the selected intrinsic reward method."""
        if self.type == 'curiosity':
            return self.curiosity
        elif self.type == 'empowerment':
            return self.empowerment
        elif self.type == 'surprise-based':
            return self.surprise_based
        else:
            raise ValueError(f"'{self.type}' does not exist! Must be one of: '{self.valid_types}'.")

    def curiosity(self) -> None:
        """
        Simulates the curiosity intrinsic reward as displayed in the X paper:
        
        """
        pass

    def empowerment(self) -> None:
        """
        Simulates the empowerment intrinsic reward as displayed in the X paper:

        """
        pass

    def surprise_based(self) -> None:
        """
        Simulates the surprised_based intrinsic reward as displayed in the X paper:

        """
        pass
