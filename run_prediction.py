# -------------------------------------------------------------------------------
# Main python run file for training a ML model, then make prediction
# -------------------------------------------------------------------------------
import os
import argparse
import json
import multiprocessing

import numpy as np
import pandas as pd

import input
import model
import training
import output
import utils

def run(params):
    # Extracting the parameter
    
    # Training the model
    training_label = f'Test {number_of_sample} varians w 0.2% sample on exp mass'

    with tf.device('/GPU:0'):
        history_1, history_2, history_3, best_weights = fine_grain_training(model, data_train, data_val, batch_number=[32, 16, 4],
                                                                        epoch_number=[250, 100, 50], training_name=training_label)

    # -------------------------------------------------------------------------------
    # Generating the mass model
    # -------------------------------------------------------------------------------
    # Re-initialize model
    model = wouter_model(N_input, 'adadelta')
    model.load_weights(best_weights)

    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = extract_varian_data(selected_varian)

    test_input, N_test_input = modified_wouter(Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian)
    generate_mass_table(model, test_input, training_label)

def main():
    # Parsing command line argument
    args = utils.run.parse_argse()

    # Load parameters from a config file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} does not exist.")
        return

    param_sets = utils.run.load_param(args.config)

    # Run the 'run' program
    if args.parallel:
        print("Running in parallel")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            run()
    else:
        print("Running in sequential")
        run()

if __name__ == "__main__":
    main()