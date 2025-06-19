import numpy as np
import pandas as pd


def extract_result(result_row):
    mass_table = load_mass_table(result_row)

    percent_error = rms(mass_table["difference"]) * 100 / rms(mass_table["target"])
    return [percent_error.mean()]


def extract_variant_data(output_file):

    return rms, std


def load_mass_table(result_row):
    mass_table = pd.read_csv(result_row["output_file"])

    return mass_table


def rms(array):
    return np.sqrt((array**2).mean())
