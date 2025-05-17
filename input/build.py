import importlib


class Input:
    """
    Initialize the Input class in the ML pipeline. This includes:
    loading the dataset, preprocessing, and feature engineering.

    Attributes:
        module (str): Python module (file) which contains the input data generator
        generator (str): Python function to generate an input data.
                         Should only take param as the input and return input data
        param (dict): Parameters to generate the input data, customize it based on the needs
        data (dict): Dictionary of training, validation, and test data (tuples)
    """

    def __init__(self, run_param) -> None:
        input_param = run_param["input"]
        self.module = input_param["module"]
        self.generator = input_param["generator"]
        self.param = input_param["param"]

    def _generate_input_data(self):
        """
        Loading the generator function and load it to the object
        """
        module = importlib.import_module(f"input.{self.module}")
        generator = getattr(module, self.generator)

        self.data = generator(self.param)

    def __getattr__(self, name):
        """

        Args:
            name (str): Attributes to be lazy loaded

        Returns:
            Attribute

        Raises:
            AttributeError: If the given attribute is not found
        """
        if name == "data":
            self._generate_input_data()
            return self.data

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
