from pathlib import Path

import pandas as pd


BSk24_dataset = {
    "BSk24_param": "bsk24.csv",
    "BSk24_mass_table": "bsk24_mass_table.csv",
    "BSk24_mass_table_exp": "bsk24_mass_table_exp.csv",
    "BSk24_variants_param": "bsk24_variants.parquet",
    "BSk24_variants_mass_table": "bsk24_variants_mass_table.parquet",
    "BSk24_variants_sample_param": "bsk24_variants_sample.csv",
    "BSk24_variants_sample_mass_table": "bsk24_variants_sample_mass_table.csv",
    "BSk24_variants_ext_param": "bsk24_variants_ext.parquet",
    "BSk24_variants_ext_mass_table": "bsk24_variants_ext_mass_table.parquet",
    "BSk24_variants_ext_sample_param": "bsk24_variants_ext_sample.parquet",
    "BSk24_variants_ext_sample_mass_table": "bsk24_variants_ext_mass_table.parquet",
}


def load_df(name):
    file = f'data/input/{BSk24_dataset[name]}'
    extension = Path(file).suffix
    print(file, extension)

    if extension == ".parquet":
        df = pd.read_parquet(file)
    elif extension == ".csv":
        df = pd.read_csv(file, sep=';')
    else:
        raise AttributeError(f"Unknown dataframe {name}")

    return df

