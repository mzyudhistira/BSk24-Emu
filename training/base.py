import time
from pathlib import Path
import shutil
import random

import numpy as np
from keras.callbacks import ModelCheckpoint, Callback


def simple(data, model, training_param, file_path):
    """
    Simple training method

    Args:
        data (dict) : Dataset used in the run
        model (model object): Model object from the pipeline
        training_param (dict): Parameter of the training
        file_path (list): Filepath of the results
    """
    best_weights, last_weights, loss_path, val_loss_path = file_path

    features, target = data["train"]
    val_features, val_target = data["val"]

    previous_weight = training_param.get("weight", "")
    if previous_weight != "":
        model.load_weights(previous_weight)

    epoch = training_param["epoch"]
    batch = training_param["batch"]

    model_checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
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


def multi_stage(data, model, training_param, file_path):
    """
    Three-step of simple training

    Args:
        data (dict) : Dataset used in the run
        model (model object): Model object from the pipeline
        training_param (dict): Parameter of the training
        file_path (list): Filepath of the results
    """
    best_weights, last_weights, loss_path, val_loss_path = file_path

    features, target = data["train"]
    val_features, val_target = data["val"]

    # Initiate param and files for each simple training
    epochs = training_param["epoch"]
    batches = training_param["batch"]

    simple_param = [
        {"epoch": epochs[0], "batch": batches[0]},
        {"weight": last_weights, "epoch": epochs[1], "batch": batches[1]},
        {"weight": best_weights, "epoch": epochs[2], "batch": batches[2]},
    ]

    simple_files = []
    cache_folder = Path("data/cache")
    cache_folder.mkdir(parents=True, exist_ok=True)

    hex_chars = "0123456789ABCDEF"
    rnd_hex = ["".join(random.choices(hex_chars, k=5)) for i in range(3)]

    for i in range(3):
        random_hex = "".join(random.choices(hex_chars, k=5))
        simple_files.append(
            [
                best_weights,
                last_weights,
                cache_folder / f"loss{i}_{rnd_hex[i]}.dat",
                cache_folder / f"val_loss{i}_{rnd_hex[i]}.dat",
            ]
        )

    # Train the modl
    for i in range(3):
        print(f"{'-'*20}")
        print(f"Training step: {i+1}")
        simple(data, model, simple_param[i], simple_files[i])

    # Combine loss and val_loss from all training steps
    loss_data = [
        np.loadtxt(file).flatten()
        for file in [cache_folder / f"loss{i}_{rnd_hex[i]}.dat" for i in range(3)]
    ]
    val_loss_data = [
        np.loadtxt(file).flatten()
        for file in [cache_folder / f"val_loss{i}_{rnd_hex[i]}.dat" for i in range(3)]
    ]

    loss_arr = [item for sublist in loss_data for item in sublist]
    val_loss_arr = [item for sublist in val_loss_data for item in sublist]
    np.savetxt(loss_path, loss_arr)
    np.savetxt(val_loss_path, val_loss_arr)

    shutil.rmtree(cache_folder)
