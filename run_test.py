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
    bsk24_param = BSk24
    bsk24_mass_table = BSk24_MASS_TABLE
    bsk24_varians_param = BSk24_VARIANS_EXT
    bsk24_varians_mass_table = BSk24_VARIANS_EXT_MASS_TABLE

    # Sampling the varian
    varian_sample = run_param[:24]
    selected_varian = select_varian(varian_sample, data="ext")
    selected_varian_sample = (
        selected_varian.groupby("varian_id", group_keys=False)
        .apply(lambda x: x.sample(frac=0.1))
        .reset_index(drop=True)
    )

    selected_varian_sample = get_mass_diff(selected_varian_sample)

    bsk24_mass_table["m_0"] = 0
    Z_bsk24 = bsk24_mass_table["Z"]
    N_bsk24 = bsk24_mass_table["N"]
    m_bsk24 = bsk24_mass_table["m_0"]
    param_bsk24 = BSk24.to_numpy()
    param_bsk24 = np.tile(param_bsk24, (len(Z_bsk24), 1))

    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = (
        extract_varian_data(selected_varian_sample)
    )

    # Generate the input date
    Z = np.concatenate((Z_bsk24, Z_bsk24_varian))
    N = np.concatenate((N_bsk24, N_bsk24_varian))
    m = np.concatenate((m_bsk24, m_bsk24_varian))
    params = np.concatenate((param_bsk24, params_bsk24_varian))

    input_data, N_input = modified_wouter(Z, N, m, params)
    # input_data, N_input = modified_wouter(
    #     Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian
    # )

    # Modify input data
    np.random.shuffle(input_data)
    data_train, data_test, data_val = split_input(input_data, 0.8, 0.1, 0.1)
    # normalize_input(data_train, data_test, data_val, input_data, N_input)

    # Initialize the model
    model = wouter_model(N_input, "rmsprop")
    model.summary()

    """
    Start training model
    """
    # Training the model
    training_label = f"EXT on diff, code:{run_param[0]}"

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
    # N_input = 31
    # best_weights = "data/training/weight_training/EXT 24 run SGD, code: 1878.batch=16.epoch=100.stage2.weights.h5"
    # training_label = f"Variant_SGD{run_param}"
    model = wouter_model(N_input, "adadelta")
    model.load_weights(best_weights)

    varian_test = run_param[24:]
    selected_varian = select_varian(varian_test, data="ext")
    selected_varian = get_mass_diff(selected_varian)
    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = (
        extract_varian_data(selected_varian)
    )
    test_input, N_test_input = modified_wouter(
        Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian
    )

    generate_mass_table(model, test_input, training_label)
    print(training_label)


def main():
    varian_ids = list(range(1, 11023))
    random.shuffle(varian_ids)
    sample_number = [24, 24, 24]
    varian_train_to_predict_ratio = 4
    test_number = [varian_train_to_predict_ratio * i for i in sample_number]

    test_param = []
    test_param.append(
        varian_ids[: sample_number[0] * (varian_train_to_predict_ratio + 1)]
    )
    test_param.append(
        varian_ids[
            sample_number[0]
            * (varian_train_to_predict_ratio + 1) : 2
            * sample_number[1]
            * (varian_train_to_predict_ratio + 1)
        ]
    )
    test_param.append(
        varian_ids[
            2
            * sample_number[1]
            * (varian_train_to_predict_ratio + 1) : 3
            * sample_number[2]
            * (varian_train_to_predict_ratio + 1)
        ]
    )

    # test_param = varian_ids[:3]

    args = parse_args()

    # Run the 'run' program
    if args.parallel:
        print("Running in parallel")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(run, test_param)
    else:
        print("Running in sequential")
        for param in test_param[:1]:
            run(param)


if __name__ == "__main__":
    main()
