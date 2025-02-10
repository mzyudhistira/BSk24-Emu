"""
Loading the libraries
"""

# External
import pandas as pd
import numpy as np

# Internal
from config import *

input_data_dir = DATA_DIR / "input"

"""
Format data to pandas readable format
"""


def format_mass_all_extrap(file):
    """
    format the extrapolated mass data, return parameter and mass table dataframe
    """
    # Extract the parameters on the top of the file
    params_up = pd.read_csv(file, nrows=44, sep=r"\s+", header=None).T
    header_up = params_up.head(1).values.tolist()[0]
    for i in range(23, 44, 3):
        header_up[i + 2] = header_up[i + 2] + "_" + header_up[i]

    params_up.columns = header_up
    params_up = params_up.drop(0)
    params_up["sampling"] = params_up["sampling"].astype(int)

    # Extract the bottom parameters
    params_dn = pd.read_csv(file, skiprows=6472, sep=r"\s+", header=None).T
    header_dn = params_dn.head(1).values.tolist()[0]
    params_dn.columns = header_dn
    params_dn = params_dn.drop(0)
    params_dn["sampling"] = params_up["sampling"].astype(int)
    params_dn = params_dn.rename(columns={"sigma": "sigma_rms"})

    # Combine both parameter dataframes
    params = pd.merge(params_up, params_dn, how="inner", on="sampling")
    params = params.rename(columns={"sampling": "varian_id"})

    # Extract the mass table
    # loss
    # deviation

    mass_table = pd.read_csv(
        ext_data_path, skiprows=46, nrows=6473 - 48, sep=r"\s+", header=None
    )
    mass_table_header = ["Z", "N", "A"] + list(range(1, 11023))
    mass_table.columns = mass_table_header
    mass_table = pd.melt(
        mass_table, id_vars=["Z", "N", "A"], var_name="varian_id", value_name="m"
    )

    # Reorder the header
    mass_table = mass_table[["varian_id", "Z", "N", "A", "m"]]

    return params, mass_table


"""
Read the data
"""


def get_bsk24():
    df = pd.read_csv(input_data_dir / "bsk24.csv", sep=";")

    return df


def get_bsk24_mass_table():
    df = pd.read_csv(input_data_dir / "bsk24_mass_table.csv", sep=";")

    return df


def get_bsk24_experimental_mass_table():
    df = pd.read_csv(input_data_dir / "bsk24_mass_table_exp.csv", sep=";")

    return df


def get_bsk24_varians(full_data=False):
    if full_data == True:
        file = "bsk24_varians.parquet"
        df = pd.read_parquet(input_data_dir / file)

    else:
        file = "bsk24_varians_sample.csv"
        df = pd.read_csv(input_data_dir / file, sep=";")

    return df


def get_bsk24_varians_mass_table(full_data=False):

    if full_data == False:
        file = "bsk24_varians_sample_mass_table.csv"
        df = pd.read_csv(input_data_dir / file, sep=";")
    else:
        file = "bsk24_varians_mass_table.parquet"
        df = pd.read_parquet(input_data_dir / file)

    return df


def get_bsk24_varians_ext():
    file = "bsk24_varians_ext_sample.parquet"
    df = pd.read_parquet(input_data_dir / file)
    df.drop("rms0", axis=1, inplace=True)

    return df


def get_bsk24_varians_ext_mass_table():
    file = "bsk24_varians_ext_sample_mass_table.parquet"
    df = pd.read_parquet(input_data_dir / file)

    return df


def read_csv(file):
    df = pd.read_csv(input_data_dir / file, sep=";")

    return df


def read_parquet(file):
    df = pd.read_parquet(input_data_dir / file)

    return df
