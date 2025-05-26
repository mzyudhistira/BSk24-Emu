import random

import numpy as np
from numpy.testing import tempdir
import pandas as pd

from . import load
from . import feature
from . import preprocess


def single_variant(param):
    """
    Generate dataset for a single variant

    Args:
        param (dict): Loaded dict from the run config

    Returns:
        data (dict): Dictionary of train, val, and test data. Each is a tuple of np arr

    """
    # Extract paremeters
    percent_train, percent_val, percent_test = param["percentage"]
    N_input = param["N_input"]
    variant_type = param["dataset"]["variant"]
    variant_percentage = param["dataset"]["variant_percentage"]

    if param["dataset"]["variant_id"] == "random":
        if variant_type == "exp":
            id_range = (1, 33220)
        elif variant_type == "ext":
            id_range = (1, 11023)
        else:
            id_range = (1, 1029)

        variant_id = random.randint(*id_range)

    else:
        variant_id = param["dataset"]["variant_id"]

    # Loading
    skyrme_param, mass_table = load.load_df("variant", variant_type)
    input_data = feature.select_variant(skyrme_param, mass_table, variant_id)

    # Feature selection
    # mass_table = feature.select_nuclei()
    input_data = load.load_ame20()
    input_tensor, N_input = feature.nuclear_properties_w(input_data)

    # Preprocessing
    np.random.shuffle(input_tensor)
    input_tensor = preprocess.normalise(input_tensor)
    data_train, data_val, data_test = preprocess.split(
        input_tensor, percent_train, percent_val, percent_test
    )
    #
    return {
        "train": extract_feature(data_train),
        "val": extract_feature(data_val),
        "test": extract_feature(data_test),
    }


def extract_feature(data):
    """
    Split an input tensor to features and target

    Args:
        data (np arr): Input tensor

    Returns:
        features, target (np arr)

    """
    return data[:, :-1], data[:, -1]
