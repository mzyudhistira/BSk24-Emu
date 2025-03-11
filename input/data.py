from pathlib import Path

import pandas as pd

from config import *

dir = DATA_DIR / "input"
BSk24_dataset = {
    "BSk24_param": dir / "bsk24.csv",
    "BSk24_mass_table_default": dir / "bsk24_mass_table.csv",
    "BSk24_mass_table_exp": dir / "bsk24_mass_table_exp.csv",
    "BSk24_varians_param_full": dir / "bsk24_varians.parquet",
    "BSk24_varians_mass_table_full": dir / "bsk24_varians_mass_table.parquet",
    "BSk24_varians_param_sample": dir / "bsk24_varians_sample.csv",
    "BSk24_varians_mass_table_sample": dir / "bsk24_varians_sample_mass_table.csv",
    "BSk24_varians_param_ext": dir / "bsk24_varians_ext.parquet",
    "BSk24_varians_mass_table_ext": dir / "bsk24_varians_ext_mass_table.parquet",
    "BSk24_varians_param_ext_sample": dir / "bsk24_varians_ext_sample.parquet",
    "BSk24_varians_mass_table_ext_sample": dir / "bsk24_varians_ext_mass_table.parquet",
}


def load_df(name):
    """Load the desired dataframe based on the dataset dictionary defined above

    Args:
        name (string): Key of the dataframe

    Raises:
        AttributeError: Unknown key of the dataframe

    Returns:
        df (pandas dataframe): dataframe of the desired data
    """
    # print(name)
    file = name
    # file = BSk24_dataset[name]
    extension = Path(file).suffix[1:]

    if extension == "parquet":
        df = pd.read_parquet(file)
    elif extension == "csv":
        df = pd.read_csv(file, sep=";")
    else:
        raise AttributeError(f"Unknown dataframe {name}")

    return df


def load_data(name):
    if name == "default" or name == "exp":
        param = load_df(BSk24_dataset["BSk24_param"])
        mass_table = load_df(BSk24_dataset["BSk24_mass_table_" + name])

    else:
        param = load_df(BSk24_dataset["BSk24_varians_param_" + name])
        mass_table = load_df(BSk24_dataset["BSk24_varians_mass_table_" + name])

    return param, mass_table
