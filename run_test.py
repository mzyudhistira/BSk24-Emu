"""
Importing required libraries
"""
#External Library
import numpy as np
import pandas as pd
from keras import models, layers, callbacks
import os
import tensorflow as tf
import sys
import ast

#Internal Library
sys.path.append('input/')
sys.path.append('model/')
sys.path.append('training/')
sys.path.append('output/')

from read_data import *
from input_data_generate import * 
from input_data_modifier import *
from model_building import *
from wouter_training import *
from model_test import *

"""
Initialization
"""
# Initialize hardware env
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Generate input data
bsk24_data = get_bsk24_data()
bsk24_variations_data = get_bsk24_varians_sample()

varian_number = 5 # Pick any number between 0 and 322118
varian = bsk24_variations_data.iloc[varian_number][:] # Pick a varian
varian_mass_table = get_bsk24_varian_mass_table(varian)

Z_bsk24 = bsk24_data['Z']
N_bsk24 = bsk24_data['A'] - Z_bsk24
m_bsk24 = bsk24_data['Mcal']

Z_bsk24_varian = varian_mass_table['Z']
N_bsk24_varian = varian_mass_table['N']
m_bsk24_varian = varian_mass_table['mass']

# Randomly select

Z = np.concat((Z_bsk24, Z_bsk24_varian))
N = np.concat((N_bsk24, N_bsk24_varian))
m = np.concat((m_bsk24, m_bsk24_varian))

input_data, N_input = generate_wouters_input_data(N, Z, m)

# Modify input data
np.random.shuffle(input_data)
N_train, N_val, N_test = [150,25,25]
data_train, data_test, data_val = select_input(input_data,N_train,N_val,N_test)
normalize_input(data_train, data_test, data_val, input_data, N_input)

# # Reorder the data
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
training_label = 'sample_test=[50,10,10]'

with tf.device('/GPU:0'):
    history_1, history_2, history_3, best_weights = fine_grain_training(model, data_train, data_val, batch_number=[32, 16, 4],
                                                                    epoch_number=[500, 100, 100], training_name=training_label)

"""
Generating mass tables
"""
# Re-initialize model
model = wouter_model(N_input, 'adadelta')
model.load_weights(best_weights)

generate_mass_table(model, input_data, training_label)
