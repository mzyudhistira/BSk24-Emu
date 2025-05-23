import numpy as np
import pandas as pd


def get_mass_diff(variant_mass_table):
    from input.data import BSk24_MASS_TABLE

    tmp_df = pd.merge(
        variant_mass_table,
        BSk24_MASS_TABLE,
        how="inner",
        on=["Z", "N"],
        suffixes=("_v", "_o"),
    )

    variant_mass_table = pd.merge(variant_mass_table, tmp_df, how="inner")
    variant_mass_table["m"] = variant_mass_table["m_v"] - variant_mass_table["m_o"]
    variant_mass_table.drop(columns=["A_v", "m_v", "A_o", "m_o"], inplace=True)
    del tmp_df

    return variant_mass_table


def normalize_params(variant_mass_table):
    from input.data import BSk24 as BSk24_param

    cname1 = [f"param(0{i})" for i in range(1, 10)]
    cname2 = [f"param({i})" for i in range(10, 22)]
    cname = cname1 + cname2

    BSk24_param.columns = cname
    intersection = variant_mass_table.columns.intersection(BSk24_param.columns)
    variant_mass_table[intersection] = (
        variant_mass_table[intersection] - BSk24_param[intersection].values
    )

    return variant_mass_table


def generate_wouters_input_data(
    N,
    Z,
    binding_energy,
    NMN=[8, 20, 28, 50, 82, 126],
    PNM=[8, 20, 28, 50, 82, 126],
    nin=10,
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
    # N_input = 10
    N_input = nin
    A = N + Z

    # Creating a complete data set from the (N,Z,BE)-data the table
    # Note that this has N_input + 1 columns: we store the binding energy here too
    # complete_dat = np.zeros((len(N), N_input + 1))
    complete_dat = np.zeros((len(N), 10 + 1))

    for i in range(len(N)):
        # - - - - - - - - - - - - - - - - - -
        # Basic information
        complete_dat[i, 0] = N[i]  # Neutron number N
        complete_dat[i, 1] = Z[i]  # Proton number Z
        complete_dat[i, 2] = (-1) ** (N[i])  # Number parity of neutrons
        complete_dat[i, 3] = (-1) ** (Z[i])  #                  protons

        # - - - - - - - - - - - - - - - - - -
        # Liquid drop parameters
        complete_dat[i, 4] = (A[i]) ** (2.0 / 3.0)  # A^2/3
        complete_dat[i, 5] = Z[i] * (Z[i] - 1) / (A[i] ** (1.0 / 3.0))  # Coulomb term
        complete_dat[i, 6] = (N[i] - Z[i]) ** 2 / A[i]  # Asymmetry
        complete_dat[i, 7] = A[i] ** (-1.0 / 2.0)  # Pairing

        # - - - - - - - - - - - - - - - -
        # Distance to the next magic number for both protons and neutrons
        dist_N = 100000
        dist_Z = 100000
        for k in NMN:
            dist_Nb = abs(N[i] - k)
            if dist_Nb < dist_N:
                dist_N = dist_Nb

        for k in PNM:
            dist_Zb = abs(Z[i] - k)
            if dist_Zb < dist_Z:
                dist_Z = dist_Zb

        complete_dat[i, 8] = dist_N
        complete_dat[i, 9] = dist_Z
        complete_dat[i, 10] = binding_energy[i]
        # - - - - - - - - - - - - - - - -
        ret = np.concatenate([complete_dat[:, :nin], complete_dat[:, -1:]], axis=1)

    # return complete_dat, N_input
    return ret, N_input


def modified_wouter(
    Z, N, M, param, NMN=[8, 20, 28, 50, 82, 126], PNM=[8, 20, 28, 50, 82, 126]
):
    """
    A modified version of wouter input data. This consists of:
    Z = Proton Number
    N = Neutron number
    param = a list of parameters that are used, this follows the order
    t_0;t_1;t_2;t_3;t_4;t_5;x_0;x_1;x_2;x_3;x_4;x_5;alpha;beta;gamma;W_0;f_n+;f_n-;f_p+;f_p-;epsilon_A
    M = mass (MeV)
    """

    wouter_dat, N_input = generate_wouters_input_data(N, Z, M, NMN=NMN, PNM=PNM)
    # N_input = 23

    complete_dat = np.concatenate(
        (wouter_dat[:, :-1], param, wouter_dat[:, -1][:, np.newaxis]), axis=1
    )
    N_input = wouter_dat[:, :-1].shape[1] + param.shape[1]

    return complete_dat, N_input
