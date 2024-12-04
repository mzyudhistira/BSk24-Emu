from keras import models, layers, Input

def wouter_model(N_input, optimizer):
    """
      Build a model that expects data of dimension N_input , that should be
      optimized with specified algorithm.

      Input:
        N_input   : dimension of the input
        optimizer : algorithm for the training
      Output:
        model : Keras model object

    """

    #define the model as a sequence of layers
    model = models.Sequential()
    model.add(Input(shape=(N_input,)))
    #first layer connected to the input
    # model.add(layers.Dense(128,activation='relu',input_shape=(N_input,)))
    model.add(layers.Dense(128,activation='relu'))
    # hidden layer
    model.add(layers.Dense(64,activation='relu'))
    #hidden layer
    model.add(layers.Dense(32, activation='relu'))
    #final layer conected to the output
    model.add(layers.Dense(1))
    #mse = mean squared error
    #mae = mean average error
    #loss : what will be minimize
    #metrics : what will be shown
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model