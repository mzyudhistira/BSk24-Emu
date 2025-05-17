import importlib
from pathlib import Path


class TrainModel:
    """
    Initialize the Train class in the ML pipeline.

    Attributes:
        module (str): Python module (file) which contains the training method
        method (str): Python procedure to run the training
        param (dict): Parameters of the training
        best_weights (str): Filepath of the saved best weights
        last_weights (str): Filepath of the saved last weights
        loss (str): Filepath of the saved loss
        val_loss (str): Filepath of the saved validation loss
    """

    def __init__(self, run_param) -> None:
        training_param = run_param["training"]
        self.module = training_param["module"]
        self.method = training_param["method"]
        self.param = training_param["param"]

        run_dir = run_param["run"]["dir"]
        self.best_weights = run_dir / "best_weights.weights.h5"
        self.last_weights = run_dir / "last_weights.weights.h5"
        self.loss = run_dir / "loss.dat"
        self.val_loss = run_dir / "val_loss.dat"

    def run_training(self, input_object, model_object):
        module = importlib.import_module(f"training.{self.module}")
        method = getattr(module, self.method)

        training_data, validation_data, _ = input_object.data
        training_result = [
            self.best_weights,
            self.last_weights,
            self.loss,
            self.val_loss,
        ]

        method(input_object.data, model_object.model, self.param, training_result)
