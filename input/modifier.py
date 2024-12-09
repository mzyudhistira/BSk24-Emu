import numpy as np
import pandas as pd

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
     train_dat : training data, first N_train rows in data
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