"""
Importing required libraries
"""
#External Library
import numpy as np
import pandas as pd
from keras import models, layers, callbacks
import os
import tensorflow as tf

#Internal Library
from input import *
from model import *
from training import *
from output import *
from utils import *

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
varian_number = 7 # Pick any number between 1 and 10
selected_varian = select_varian(varian_number)
selected_varian_sample = selected_varian.sample(frac=0.2).reset_index(drop=True)

Z_bsk24 = bsk24_mass_table['Z']
N_bsk24 = bsk24_mass_table['N']
m_bsk24 = bsk24_mass_table['m']
param_bsk24 = BSk24.to_numpy()
param_bsk24 = np.tile(param_bsk24, (len(Z_bsk24), 1))

Z_bsk24_varian = selected_varian_sample['Z']
N_bsk24_varian = selected_varian_sample['N']
m_bsk24_varian = selected_varian_sample['m']
params_bsk24_varian = selected_varian_sample.iloc[:,5:26].to_numpy()

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
training_label = 'test_var7_unscaledInput'

with tf.device('/GPU:0'):
    history_1, history_2, history_3, best_weights = fine_grain_training(model, data_train, data_val, batch_number=[32, 16, 4],
                                                                    epoch_number=[250, 100, 50], training_name=training_label)

"""
Generating mass tables
"""
# Re-initialize model
model = wouter_model(N_input, 'adadelta')
model.load_weights(best_weights)

Z_bsk24_varian = selected_varian['Z']
N_bsk24_varian = selected_varian['N']
m_bsk24_varian = selected_varian['m']
params_bsk24_varian = selected_varian.iloc[:,5:26].to_numpy()

test_input, N_test_input = modified_wouter(Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian)
generate_mass_table(model, test_input, training_label)
