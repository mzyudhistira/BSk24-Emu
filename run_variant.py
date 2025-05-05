import os
import random

import numpy as np
import pandas as pd

from input import *
from model import *
from training import *
from output import *
from utils import *
from run_analysis import plot_loss


def run(param):
    # Initialize hardware env
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Extract parameters
    variant_sample, neurons, label = param

    # Generate input data
    bsk24_param = BSk24
    bsk24_mass_table = BSk24_MASS_TABLE
    bsk24_variants_param = BSk24_VARIANS_EXT[
        BSk24_VARIANS_EXT["varian_id"] == variant_sample
    ]
    bsk24_variants_mass_table = BSk24_VARIANS_EXT_MASS_TABLE[
        BSk24_VARIANS_EXT_MASS_TABLE["varian_id"] == variant_sample
    ]

    selected_variant_sample = (
        bsk24_variants_mass_table.groupby("varian_id", group_keys=False)
        .apply(lambda x: x.sample(frac=0.4))
        .reset_index(drop=True)
    )

    z = selected_variant_sample["Z"].values[:, None]
    n = selected_variant_sample["N"].values[:, None]
    m = selected_variant_sample["m"].values[:, None]
    input_data = np.concatenate((n, z, m), axis=1)
    N_input = 2

    np.random.shuffle(input_data)
    data_train, data_test, data_val = split_input(input_data, 0.8, 0.1, 0.1)
    # normalize_input(data_train, data_test, data_val, input_data, N_input)

    # Initialize the model
    model = sequential_model(N_input, neurons)
    model.summary()

    # Start training
    training_label = label
    training_batches = [32, 16, 4]
    training_epochs = [1000, 200, 50]
    with tf.device("/GPU:0"):
        history_1, history_2, history_3, best_weights = fine_grain_training(
            model,
            data_train,
            data_val,
            batch_number=training_batches,
            epoch_number=training_epochs,
            training_name=training_label,
        )

    # Make prediction
    model = sequential_model(N_input, neurons)
    model.load_weights(best_weights)

    selected_varian = select_varian(variant_sample, data="ext")
    z = selected_variant_sample["Z"].values[:, None]
    n = selected_variant_sample["N"].values[:, None]
    m = selected_variant_sample["m"].values[:, None]

    test_input = np.concatenate((n, z, m), axis=1)
    # normalise(test_input)

    result = generate_mass_table(model, test_input, training_label)
    rms_dev = np.sqrt((result["Difference"] ** 2).mean())
    avg_dev = result["Difference"].mean()
    std_diff = result["Difference"].std()

    loss_file = (
        "data/training/loss/"
        + training_label
        + f".batch={training_batches[-1]}.epoch={training_epochs[-1]}.stage3.loss.dat"
    )
    with open(loss_file, "r") as file:
        last_loss = float(file.readlines()[-1].strip())

    return rms_dev, avg_dev, last_loss


def main():
    variant = 3123
    neurons = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    labels = [f"Variant {variant} with arch_id {x}" for x in range(1, len(neurons) + 1)]
    observed_effect = []

    for i in range(len(neurons)):
        test_param = [variant, neurons[i], labels[i]]
        effect = run(test_param)
        observed_effect.append(effect)

    observed_effect = np.array(observed_effect)
    df = pd.DataFrame(
        {
            "variant": [variant] * len(neurons),
            "arch_id": list(range(1, len(neurons) + 1)),
            "neurons": neurons,
            "rms_dev": observed_effect[:, 0],
            "avg_dev": observed_effect[:, 1],
            "last_loss": observed_effect[:, 2],
        }
    )

    df.to_csv("data/output/250505/neurons_effect_custom.csv", index=False)


if __name__ == "__main__":
    main()
