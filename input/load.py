from pathlib import Path

import pandas as pd

from input.feature import skyrme_param


BSk24_files = {
    "skyrme_param": {
        "reg_full": "bsk24.csv",
        "reg_exp": "bsk24.csv",
        "variant_exp": "bsk24_variants.parquet",
        "variant_exp_sample": "bsk24_variants_sample.csv",
        "variant_ext": "bsk24_variants_ext.parquet",
        "variant_ext_sample": "bsk24_variants_ext_sample.parquet",
    },
    "mass_table": {
        "reg_full": "bsk24_mass_table.csv",
        "reg_exp": "bsk24_mass_table_exp.csv",
        "variant_exp": "bsk24_variants_mass_table.parquet",
        "variant_exp_sample": "bsk24_variants_sample_mass_table.csv",
        "variant_ext": "bsk24_variants_ext_mass_table.parquet",
        "variant_ext_sample": "bsk24_variants_ext_sample_mass_table.parquet",
    },
}


def load_ame20():
    df = pd.read_csv("data/input/ame20.dat", delim_whitespace=True)
    df["A"] = df["Z"] + df["N"]
    df["m"] = df["BE"]

    return df


def load_df(dataset, type):
    """
    Load BSk24 dataset

    Args:
        dataset (string): Type of BSk24 data: reg or variant
        type (str): Type of the dataset. Possible values:
                    reg
                    - full : Full mass
                    - exp : Experimental mass

                    variant
                    - exp : Experimental mass
                    - exp_sample : Sample of the experimental mass, can be used to reduce memory usage
                    - ext : All extrapolated mass
                    - ext_sample : Sample of all extrapolated mass, can be used to reduce memory usage

    Returns:
        res (list): List of [skyrme_param(pd df), mass_table(pd df)]

    Raises:
        AttributeError:
    """
    root_file = Path("data/input")
    skyrme_param_file = root_file / BSk24_files["skyrme_param"][f"{dataset}_{type}"]
    mass_table_file = root_file / BSk24_files["mass_table"][f"{dataset}_{type}"]

    files = [skyrme_param_file, mass_table_file]
    res = []

    for file in files:
        extension = Path(file).suffix

        if extension == ".parquet":
            res.append(pd.read_parquet(file))
        elif extension == ".csv":
            res.append(pd.read_csv(file, sep=";"))
        else:
            raise AttributeError(f"Unknown dataset {dataset} or type {type}")

    return res
