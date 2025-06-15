import numpy as np
import pandas as pd

from . import load


def select_variant(skyrme_param_dataset, mass_table, variant_id):
    """
    Select specific variant(s) form the dataset

    Args:
        skyrme_param_dataset (pd df): Dataframe loaded from load_df
        mass_table (pd df): Dataframe loaded from load_df
        variant_id (int/list of int): Id of the desired variant

    Returns:
        selected_variant (pd df) : Merged param and mass table for the selected variant(s)

    """
    if isinstance(variant_id, int):
        variant_id = [variant_id]

    variant_list = pd.DataFrame({"varian_id": variant_id})

    skyrme_param_dataset = pd.merge(
        skyrme_param_dataset, variant_list, on="varian_id", how="inner"
    )
    mass_table = pd.merge(mass_table, variant_list, on="varian_id", how="inner")

    selected_variant = pd.merge(mass_table, skyrme_param_dataset, on="varian_id")

    # Fix varian parameter x_2 to t_2.x_2
    selected_variant.iloc[:, 13] = (
        selected_variant.iloc[:, 7] * selected_variant.iloc[:, 13]
    )

    return selected_variant


def select_nuclei(mass_table, kind):
    if kind == "even":
        pass

    elif kind == "odd":
        pass

    elif kind == "even-odd":
        pass
    else:
        raise

    return


def rm_Z81(mass_table):
    mass_table = mass_table[~((mass_table["Z"] == 81) & (mass_table["N"] > 155))]

    return mass_table


def nuclear_properties(
    input_data, N_input=10, NMN=[8, 20, 28, 50, 82, 126], PMN=[8, 20, 28, 50, 82, 126]
):
    """
    Generate nuclear properties features from a given mass table

    Args:
        input_data (pd df): Dataframe of merged param and mass_table
        N_input (int): Number of properties used, default to 10
        NMN (list): List of Neutron magic number
        PMN (list): List of Proton magic number

    Returns:
        returned_dat (np arr): Input tensor, the size depends on the given N_input

    """
    # Extract data
    N = input_data["N"].to_numpy()
    Z = input_data["Z"].to_numpy()
    A = input_data["A"].to_numpy()
    m = input_data["m"].to_numpy()

    NMN = np.array(NMN)
    PNM = np.array(PMN)

    # Building the input tensor
    n_of_rows = N.shape[0]
    complete_dat = np.zeros((n_of_rows, 11))

    # Basic nuclear information
    complete_dat[:, 0] = N  # Neutron Number
    complete_dat[:, 1] = Z  # Proton Number
    complete_dat[:, 2] = (-1) ** N  # Number parity of neutrons
    complete_dat[:, 3] = (-1) ** Z  # Number parity of protons

    # Liquid drop parameters
    complete_dat[:, 4] = (A) ** (2.0 / 3.0)  # A^2/3
    complete_dat[:, 5] = Z * (Z - 1) / (A ** (1.0 / 3.0))  # Coulomb term
    complete_dat[:, 6] = (N - Z) ** 2 / A  # Asymmetry
    complete_dat[:, 7] = A ** (-1.0 / 2.0)  # Pairing

    # Distance to the next magic number for both protons and neutrons
    dist_N = N[:, None] - NMN
    dist_Z = Z[:, None] - PMN

    complete_dat[:, 8] = abs(dist_N).min(axis=1)
    complete_dat[:, 9] = abs(dist_Z).min(axis=1)

    ## Mass
    complete_dat[:, 10] = m

    returned_dat = np.concatenate(
        [complete_dat[:, :N_input], complete_dat[:, -1:]], axis=1
    )

    return returned_dat


def relative_mass(variant_mass_table):
    """
    Change the mass table's mass to be the mass difference wrt BSk24

    Args:
        variant_mass_table (pd df): Mass table of the variant(s)

    Returns:

    """
    _, bsk_mass_table = load.load_df("reg", "full")
    tmp_df = pd.merge(
        variant_mass_table,
        bsk_mass_table,
        how="inner",
        on=["Z", "N"],
        suffixes=("_v", "_o"),
    )

    variant_mass_table = pd.merge(variant_mass_table, tmp_df, how="inner")
    variant_mass_table["m"] = variant_mass_table["m_v"] - variant_mass_table["m_o"]
    variant_mass_table.drop(columns=["A_v", "m_v", "A_o", "m_o"], inplace=True)

    return variant_mass_table


def skyrme_param():
    return
