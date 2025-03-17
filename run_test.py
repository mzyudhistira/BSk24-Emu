"""
Importing required libraries
"""

# External Library
import gc
import numpy as np
import pandas as pd
from keras import models, layers, callbacks
import os
import tensorflow as tf
import random

# Internal Library
from input import *
from model import *
from training import *
from output import *
from utils import *


def run(run_param):
    """
    Initialization
    """
    # Initialize hardware env
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Generate input data
    bsk24_param, bsk24_mass_table = load_data("default")
    bsk24_varians_param, bsk24_varians_mass_table = load_data("ext")

    # Sampling the varian
    number_of_sample = run_param
    varian_numbers = random.sample(range(1, 11022), number_of_sample)
    # varian_numbers = random.sample(range(1, 10), number_of_sample)

    selected_varian = select_varian(varian_numbers, "ext")
    selected_varian_sample = (
        selected_varian.groupby("varian_id", group_keys=False)
        .apply(lambda x: x.sample(frac=0.2))
        .reset_index(drop=True)
    )

    Z_bsk24 = bsk24_mass_table["Z"]
    N_bsk24 = bsk24_mass_table["N"]
    m_bsk24 = bsk24_mass_table["m"]
    param_bsk24 = bsk24_param.to_numpy()
    param_bsk24 = np.tile(param_bsk24, (len(Z_bsk24), 1))

    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = (
        extract_varian_data(selected_varian_sample)
    )

    # Generate the input data
    Z = np.concatenate((Z_bsk24, Z_bsk24_varian))
    N = np.concatenate((N_bsk24, N_bsk24_varian))
    m = np.concatenate((m_bsk24, m_bsk24_varian))
    params = np.concatenate((param_bsk24, params_bsk24_varian))

    # input_data, N_input = generate_wouters_input_data(Z_bsk24, N_bsk24, m_bsk24)
    input_data, N_input = modified_wouter(Z, N, m, params)

    # Modify input data
    np.random.shuffle(input_data)
    data_train, data_test, data_val = split_input(input_data, 0.8, 0.1, 0.1)
    normalize_input(data_train, data_test, data_val, input_data, N_input)

    # Initialize the model
    model = wouter_model(N_input, "rmsprop")
    model.summary()

    """
    Start training model
    """
    # Training the model
    varian_train_to_predict_ratio = 2
    training_label = f"AAA Sample test"

    with tf.device("/GPU:0"):
        history_1, history_2, history_3, best_weights = fine_grain_training(
            model,
            data_train,
            data_val,
            batch_number=[32, 16, 4],
            epoch_number=[100, 50, 10],
            training_name=training_label,
        )

    """
    Generating mass tables
    """
    # Re-initialize model
    model = wouter_model(N_input, "adadelta")
    model.load_weights(best_weights)

    number_of_sample = run_param
    varian_number = random.sample(
        range(1, 11022), number_of_sample * varian_train_to_predict_ratio
    )
    selected_varian = select_varian(varian_number, "ext")
    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = (
        extract_varian_data(selected_varian)
    )

    test_input, N_test_input = modified_wouter(
        Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian
    )

    generate_mass_table(model, test_input, training_label)


def chunk_list(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def test_load():
    return


def main():
    args = parse_args()

    # Run the 'run' program
    if args.parallel:
        print("Running in parallel")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(run, test_param)
    else:
        print("Running in sequential")
        run(3)
        gc.collect()


if __name__ == "__main__":
    main()
