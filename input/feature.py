import numpy as np
import pandas as pd


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
    complete_dat = np.zeros((n_of_rows, N_input + 1))

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

    complete_dat[:, 8] = dist_N.min(axis=1)
    complete_dat[:, 9] = dist_Z.min(axis=1)

    ## Mass
    complete_dat[:, 10] = m

    returned_dat = np.concatenate(
        [complete_dat[:, :N_input], complete_dat[:, -1:]], axis=1
    )

    return returned_dat


def nuclear_properties_w(
    df, NMN=[8, 20, 28, 50, 82, 126], PNM=[8, 20, 28, 50, 82, 126]
):
    """
    Generate a complete set of nuclear data from the nuclear masses.
    Requires the file 'masses_ZN_shuf.dat' which contains in random order
        Z  N  BE(Z,N)
    This routine will generate a numpy array with 11 columns:

      0  1  2   3    4      5        6          7      8      9     10
      N  Z  PiN PiZ  A^2/3  Coul  (N-Z)**2/A  A^(-1/2)distN distZ  BE(Z,N)

    where
      Coul = Z * (Z-1)/(A**(1./3.))
      distN= distance to nearest neutron magic number, selected from input
             array NMN
      distZ= same but for proton magic numbers from PNM

    --------------------------------------------------------------------------

    Input :
      NMN, PNM: list of neutron and proton magic numbers
    Output:
      complete_dat : numpy array as described above
      N_input      : column dimension of complete_dat

    """

    # Size of the input, i.e. the number of parameters passed in to the MLNN for
    # any given input
    N_input = 10

    dat = df

    # Creating a complete data set from the (N,Z,BE)-data the table
    # Note that this has N_input + 1 columns: we store the binding energy here too
    complete_dat = np.zeros((len(dat.iloc[:, 0]), N_input + 1))

    for i in range(len(dat.iloc[:, 0])):

        Z = dat["Z"].iloc[i]
        N = dat["N"].iloc[i]
        A = dat["A"].iloc[i]

        # - - - - - - - - - - - - - - - - - -
        # Basic information
        complete_dat[i, 0] = N  # Neutron number N
        complete_dat[i, 1] = Z  # Proton number Z
        complete_dat[i, 2] = (-1) ** (N)  # Number parity of neutrons
        complete_dat[i, 3] = (-1) ** (Z)  #                  protons

        # - - - - - - - - - - - - - - - - - -
        # Liquid drop parameters
        complete_dat[i, 4] = (A) ** (2.0 / 3.0)  # A^2/3
        complete_dat[i, 5] = Z * (Z - 1) / (A ** (1.0 / 3.0))  # Coulomb term
        complete_dat[i, 6] = (N - Z) ** 2 / A  # Asymmetry
        complete_dat[i, 7] = A ** (-1.0 / 2.0)  # Pairing

        # - - - - - - - - - - - - - - - -
        # Distance to the next magic number for both protons and neutrons
        dist_N = 100000
        dist_Z = 100000
        for k in NMN:
            dist_Nb = abs(N - k)
            if dist_Nb < dist_N:
                dist_N = dist_Nb

        for k in PNM:
            dist_Zb = abs(Z - k)
            if dist_Zb < dist_Z:
                dist_Z = dist_Zb

        complete_dat[i, 8] = dist_N
        complete_dat[i, 9] = dist_Z
        # The binding energy
        complete_dat[i, 10] = dat["BE"].iloc[i]
        # - - - - - - - - - - - - - - - -

    return complete_dat, N_input

def skyrme_param():
    return
