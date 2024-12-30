import numpy as np
import pandas as pd

from input.data import *

def select_input(data, N_train, N_val, N_test):
  """
   Select from a complete set of data, a set of
     (1) training data
     (2) validation data
     (3) testing data

   If data is in random order, then all subsets of ddata will be too.

   Input:
     data   : array to chop up into parts
     N_train: number of training data
     N_val  : number of validation data
     N_test : number of testing data

   Output:

     test_dat  : testing data, last N_test rows in data
     val_dat   : validation data

  """

  val_dat   = np.copy(data[-N_val-N_test:-N_test,:])
  test_dat  = np.copy(data[-N_test:,:])
  train_dat = np.copy(data[:N_train,:])
 
  return train_dat, test_dat, val_dat

  
def normalize_input(train_dat, test_dat, val_dat, complete_dat, N_input):
  """
    Normalize all input data, i.e. for all input columns do
    
      column = (column - mean)/std
   
    where mean is the mean over the column and std is the standard deviation.
  
  """

  for i in range(N_input):
    # Subtract mean
    mean = train_dat[:,i].mean(axis=0)
    train_dat[:,i] = train_dat[:,i] - mean
    test_dat[:,i]  = test_dat[:,i] - mean
    val_dat[:,i]   = val_dat[:,i] - mean
    complete_dat[:,i]   = complete_dat[:,i] - mean
    
    # Divide by standard-deviation
    std = train_dat[:,i].std(axis=0)
    train_dat[:,i] = train_dat[:,i] /std 
    test_dat[:,i]  = test_dat[:,i] /std 
    val_dat[:,i]   = val_dat[:,i] /std 
    complete_dat[:,i]   = complete_dat[:,i] /std

def reorder_data(N, Z, data):
  sorted_indices = np.lexsort((Z, N))
  sorted_data = data[sorted_indices]

  return sorted_data

def split_input(data, percent_train, percent_val, percent_test):
  """
  Split the given input data into train, val, and test data using the provided percentage for each subset.
  """
  percentages = [percent_train, percent_val, percent_test]
  total_length = len(data)

  indices = [int(total_length * sum(percentages[:i])) for i in range(1, len(percentages))]
  split_arrays = np.split(data, indices)  
  
  return split_arrays[0], split_arrays[1], split_arrays[2]

def extract_varian_data(df):
  """
  Extracting the Z, N, mass, and skyrme parameters from a varian dataframe
  """

  Z = df['Z']
  N = df['N']
  mass = df['m']
  params = df.iloc[:,5:26].to_numpy()

  return Z, N, mass, params

def select_varian(varian_number):
    """
    Select one varian from the whole BSk24 variations
    """
    varian_param = BSk24_VARIANS[BSk24_VARIANS['varian_id']==varian_number]
    varian_mass_table = BSk24_VARIANS_MASS_TABLE[BSk24_VARIANS_MASS_TABLE['varian_id']==varian_number]
    selected_varian = pd.merge(varian_mass_table, varian_param, on='varian_id')
    
    # Fix varian parameter x_2 to t_2.x_2
    selected_varian.iloc[:,13] = selected_varian.iloc[:,7] * selected_varian.iloc[:,13]

    return selected_varian
