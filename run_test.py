"""
Importing required libraries
"""
#External Library
import gc
import numpy as np
import pandas as pd
from keras import models, layers, callbacks
import os
import tensorflow as tf
import random

#Internal Library
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
    bsk24_mass_table = BSk24_EXPERIMENTAL_MASS_TABLE
    bsk24_varians_param = BSk24_VARIANS
    bsk24_varians_mass_table = BSk24_VARIANS_MASS_TABLE

    # Sampling the varian
    # varian_number = [1,2,4,5,7,8,9,10] # Pick any number between 1 and 10
    number_of_sample = run_param
    varian_number = random_integers = random.sample(range(1, 32120), number_of_sample)
    selected_varian = select_varian(varian_number)
    print(selected_varian.shape)
    # selected_varian_sample = selected_varian.sample(frac=0.2).reset_index(drop=True)
    # -------------------------------------------------------------------------------
    selected_varian_sample = selected_varian.groupby('varian_id', group_keys=False).apply(
                            lambda x: x.sample(frac=0.2)).reset_index(drop=True)


    Z_bsk24 = bsk24_mass_table['Z']
    N_bsk24 = bsk24_mass_table['N']
    m_bsk24 = bsk24_mass_table['m']
    param_bsk24 = BSk24.to_numpy()
    param_bsk24 = np.tile(param_bsk24, (len(Z_bsk24), 1))

    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = extract_varian_data(selected_varian_sample)

    # Generate the input date
    Z = np.concatenate((Z_bsk24, Z_bsk24_varian))
    N = np.concatenate((N_bsk24, N_bsk24_varian))
    m = np.concatenate((m_bsk24, m_bsk24_varian))
    params = np.concatenate((param_bsk24, params_bsk24_varian))

    input_data, N_input = modified_wouter(Z, N, m, params)
    # input_data, N_input = generate_wouters_input_data(Z_bsk24, N_bsk24, m_bsk24)

    # Modify input data
    np.random.shuffle(input_data)
    # N_train, N_val, N_test = [150,25,25]
    # data_train, data_test, data_val = select_input(input_data,N_train,N_val,N_test)
    data_train, data_test, data_val = split_input(input_data, 0.8, 0.1, 0.1)
    # normalize_input(data_train, data_test, data_val, input_data, N_input)

    # Reorder the data
    # data_train = reorder_data(data_train[:,0], data_train[:,1], data_train)
    # data_test = reorder_data(data_test[:,0], data_test[:,1], data_test)
    # data_val = reorder_data(data_val[:,0], data_val[:,1], data_val)

    # Initialize the model
    model = wouter_model(N_input, 'rmsprop')
    model.summary()

    '''
    Start training model
    '''
    # Training the model
    varian_train_to_predict_ratio = 8
    training_label = f'Test {number_of_sample} varians w 0.2% sample each on exp mass predict {varian_train_to_predict_ratio*number_of_sample} random ext samples'

    with tf.device('/GPU:0'):
        history_1, history_2, history_3, best_weights = fine_grain_training(model, data_train, data_val, batch_number=[32, 16, 4],
                                                                        epoch_number=[1000, 500, 50], training_name=training_label)

    """
    Generating mass tables
    """
    # Re-initialize model
    gc.collect()
    model = wouter_model(N_input, 'adadelta')
    model.load_weights(best_weights)

    number_of_sample = run_param
    varian_number = random_integers = random.sample(range(1, 2500), number_of_sample * varian_train_to_predict_ratio)
    selected_varian = select_varian(varian_number, data='ext')
    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = extract_varian_data(selected_varian)

    test_input, N_test_input = modified_wouter(Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian)
    generate_mass_table(model, test_input, training_label)

def main():
    test_param = [2,4,8,16,32,64]
    args = parse_args()

    # Run the 'run' program
    if args.parallel:
        print("Running in parallel")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(run, test_param)
    else:
        print("Running in sequential")
        for param in test_param:
            run(param)

if __name__ == "__main__":
    main()