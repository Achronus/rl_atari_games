

class MissingVariableError(Exception):
    def __init__(self, message: str = "Variable is missing!") -> None:
        self.message = message
        super().__init__(self.message)


class MissingCheckpointKeyError(Exception):
    def __init__(self, message: str = 'Key is missing!') -> None:
        self.message = message
        super().__init__(self.message)


class InvalidModelTypeError(Exception):
    def __init__(self, message: str = 'Invalid model type!') -> None:
        self.message = message
        super().__init__(self.message)
