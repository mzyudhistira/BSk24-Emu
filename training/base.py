import time
from pathlib import Path

import numpy as np
from keras.callbacks import ModelCheckpoint, Callback

from utils.string import generate_random_hex


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
    # Extract variables
    best_weights, last_weights, loss_path, val_loss_path = file_path

    features, target = data["train"]
    val_features, val_target = data["val"]
    epochs = training_param["epoch"]
    batches = training_param["batch"]

    # Initiate param and files for each simple training
    cache_folder = Path("data/cache")
    simple_files = []

    rnd_hex = [
        generate_random_hex() for i in range(3)
    ]  # Random hex for temporary file to avoid overwriting files on parallel run

    simple_param = [
        {"epoch": epochs[0], "batch": batches[0]},
        {"weight": last_weights, "epoch": epochs[1], "batch": batches[1]},
        {"weight": best_weights, "epoch": epochs[2], "batch": batches[2]},
    ]

    best_weight_file = [
        cache_folder / f"best_weights_{i}_{rnd_hex[i]}.weights.h5" for i in range(2)
    ]
    best_weight_file.append(best_weights)

    for i in range(3):
        simple_files.append(
            [
                best_weight_file[i],
                last_weights,
                cache_folder / f"loss{i}_{rnd_hex[i]}.dat",
                cache_folder / f"val_loss{i}_{rnd_hex[i]}.dat",
            ]
        )

    # Train the model
    for i in range(3):
        print(f"{'-'*20}")
        print(f"Training step: {i+1}")

        if i == 2:
            # Find the training step with lowest val_loss, then load the corresponding weight
            val_loss_old = np.loadtxt(simple_files[0][3])
            val_loss_new = np.loadtxt(simple_files[1][3])

            if min(val_loss_old) < min(val_loss_new):
                print("-" * 20)
                print("old is better")
                simple_param[2]["weight"] = best_weight_file[0]

            else:
                simple_param[2]["weight"] = best_weight_file[1]

            # Change model's optimizer
            model.compile(optimizer="adagrad", loss="mse")

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

