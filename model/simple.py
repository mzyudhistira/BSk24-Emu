import tensorflow as tf
from keras import models, layers, Input


def sequential(model_param):
    """
    Generate a simple sequential neural netwok

    Args:
        model_param(dict):
            N_input (int): The number of input data
            neurons (list): A list of neurons for every hidden layers
            activation_function (keras activation function): Activation function for hidden layers, defaults to 'relu'
            optimizer (keras optimizer): Optimzier for the training, defaults to 'rmsprop'
            loss (keras loss): Loss function, defaults to 'mse'
            metrics (keras metrics): Metrics to be shown during training, defaults to ['mae']

    Returns:
        model (keras model)
    """
    # Get the attributes, set them to defaults if None is given
    activation_function = model_param.get("activation_function", "relu")
    optimizer = model_param.get("optimizer", "rmsprop")
    loss = model_param.get("loss", "mse")
    metrics = model_param.get("metrics", ["mae"])

    # Initialize the model
    model = models.Sequential()
    model.add(Input(shape=(model_param["N_input"],)))

    for neuron in model_param["neurons"]:
        model.add(layers.Dense(neuron, activation=activation_function))

    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def sequential_dropout(model_param):
    """
    Generate a simple sequential neural netwok

    Args:
        model_param(dict):
            N_input (int): The number of input data
            neurons (list): A list of neurons for every hidden layers
            dropout_rate (float or list): Dropout rate of the neurons in the hidden layers, defaults to 0.3
            activation_function (keras activation function): Activation function for hidden layers, defaults to 'relu'
            optimizer (keras optimizer): Optimzier for the training, defaults to 'rmsprop'
            loss (keras loss): Loss function, defaults to 'mse'
            metrics (keras metrics): Metrics to be shown during training, defaults to ['mae']

    Returns:
        model (keras model)
    """
    # Get the attributes, set them to defaults if None is given
    activation_function = model_param.get("activation_function", "relu")
    optimizer = model_param.get("optimizer", "rmsprop")
    loss = model_param.get("loss", "mse")
    metrics = model_param.get("metrics", ["mae"])
    dropout_rate = model_param.get("dropout_rate", 0.3)

    if isinstance(dropout_rate, float):
        dropout_rate = [dropout_rate] * len(model_param["neurons"])

    elif isinstance(dropout_rate, list):
        if len(dropout_rate) != len(model_param["neurons"]):
            raise ValueError(
                "If dropout rate is a list, the length must match the one of neurons"
            )

    # Initialize the model
    model = models.Sequential()
    model.add(Input(shape=(model_param["N_input"],)))

    for neuron, d_rate in zip(model_param["neurons"], dropout_rate):
        model.add(layers.Dense(neuron, activation=activation_function))
        model.add(layers.Dropout(d_rate))

    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
