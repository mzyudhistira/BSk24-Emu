import numpy as np


def split(data, N_train, N_val, N_test):
    """
    Split a given dataset into training, validation, and test data

    Args:
        data (np array): Dataset to split
        N_train (float): Percentage of training data
        N_val (float): Percentage of validation data
        N_test (float): Percentage of test data

    Returns:
       data_train,  data_val, data_test (np array)
    """

    percentages = [N_train, N_val, N_test]
    total_length = len(data)

    indices = [
        int(total_length * sum(percentages[:i])) for i in range(1, len(percentages))
    ]
    split_arrays = np.split(data, indices)

    return split_arrays[0], split_arrays[1], split_arrays[2]


def normalise(data):
    """
    Normalise each column of the dataset as data = (data - mean)/std

    Args:
        data (np array): input data

    Returns:
        normalised_data (np array): normalised data
    """
    feature = data[:, :-1]
    target = data[:, -1]

    mean = np.mean(feature, axis=0)
    std = np.std(feature, axis=0)
    feature = (feature - mean) / std

    normalised_data = np.column_stack((feature, target))

    return normalised_data
