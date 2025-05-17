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
