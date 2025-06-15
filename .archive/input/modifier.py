import numpy as np
import pandas as pd

from input.data import *


def extract_varian_data(df):
    """
    Extracting the Z, N, mass, and skyrme parameters from a varian dataframe
    """

    Z = df["Z"]
    N = df["N"]
    mass = df["m"]
    params = df.iloc[:, 5:26].to_numpy()

    return Z, N, mass, params
