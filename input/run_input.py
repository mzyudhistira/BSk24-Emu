import importlib

import numpy as np
import pandas as pd

from data import load_data
from modifier import extract_varian_data, select_varian
from utils.run import command


def main(input):
    """Initializing the input data in the machine learning pipeline

    Args:
        input (dictionary): Dictionary from the input file

    Returns:
        input_data (np.array): A tensor used as an input data
    """

    BSk24_param, BSk24_mass_table = load_data(input["BSk24"])
    BSk24_varian_param, BSk24_varians_mass_table = load_data(input["BSk24_variant"])

    number_of_sample = input["input_vector"]["sample_number"]
    total_variants = BSk24_varian_param["varian_id"].nunique()
    varian_number = random.sample(range(1, total_variants + 1), number_of_sample)
    selected_varian = select_varian(varian_number, data="ext")

    Z_bsk24 = BSk24_mass_table["Z"]
    N_bsk24 = BSk24_mass_table["N"]
    m_bsk24 = BSk24_mass_table["m"]
    param_bsk24 = BSk24_param.to_numpy()
    param_bsk24 = np.tile(param_bsk24, (len(Z_bsk24), 1))

    Z_bsk24_varian, N_bsk24_varian, m_bsk24_varian, params_bsk24_varian = (
        extract_varian_data(selected_varian)
    )

    Z = np.concatenate((Z_bsk24, Z_bsk24_varian))
    N = np.concatenate((N_bsk24, N_bsk24_varian))
    m = np.concatenate((m_bsk24, m_bsk24_varian))
    params = np.concatenate((param_bsk24, params_bsk24_varian))
    main_vector = {"Z": Z, "N": N, "m": m, "params": params}

    input_module = importlib.import_module(input["input_module"])
    input_data = input_module(main_vector, input["input_vector"])
    np.random.shuffle(input_data)

    return input_data


if __name__ == "__main__":
    input_file = command(sys.argv)
    main(input_file)
