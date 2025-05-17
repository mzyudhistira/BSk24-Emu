import importlib


class Model:
    """
    Initialize the Model class in the ML pipeline.

    Attributes:
        module (str): Python module (file) which contains the model generator
        generator (str): Python function to generate model.
                         Should only take param as input and return model
        param (dict): Parameters to generate the model
        model (keras model): Generated model
    """

    def __init__(self, input_object, run_param) -> None:
        model_param = run_param["model"]
        self.module = model_param["module"]
        self.generator = model_param["generator"]
        self.param = model_param["param"]

        # Loading the number of input neurons
        N_input = input_object.data["train"][0].shape[1]
        model_param["param"]["N_input"] = N_input

    def _generate_model(self):
        """
        Loading the generator function and load it to the object
        """
        module = importlib.import_module(f"model.{self.module}")
        generator = getattr(module, self.generator)

        self.model = generator(self.param)

    def __getattr__(self, name):
        """

        Args:
            name (str): Attributes to be lazy loaded

        Returns:
            Attribute

        Raises:
            AttributeError: If the given attribute is not found
        """
        if name == "model":
            self._generate_model()
            return self.model

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
