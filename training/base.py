import time

import numpy as np
from keras.callbacks import ModelCheckpoint, Callback


def simple(data, model, training_param, file_path):
    """
    Simple training method

    Args:
        data (dict) : Dataset used in the run
        model (model object): Model object from the pipeline
        file_path (list): Filepath of the results
    """
    best_weights, last_weights, loss_path, val_loss_path = file_path

    features, target = data["train"]
    val_features, val_target = data["val"]

    epoch = training_param["epoch"]
    batch = training_param["batch"]

    model_checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_best_only=True,
        monitor="val_mae",
        mode="min",
        filepath=best_weights,
    )

    history = model.fit(
        features,
        target,
        validation_data=(val_features, val_target),
        batch_size=batch,
        epochs=epoch,
        verbose=2,
        callbacks=[model_checkpoint_callback],
    )

    model.save_weights(last_weights)
    np.savetxt(loss_path, history.history["loss"])
    np.savetxt(val_loss_path, history.history["val_loss"])


def multi_stage():

    return
