

class MissingVariableError(Exception):
    def __init__(self, message: str = "Variable is missing!") -> None:
        self.message = message
        super().__init__(self.message)
