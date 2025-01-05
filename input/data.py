import os
from pathlib import Path

from .generator import *
from .modifier import *
from .reader import *

# Initializing data
BSk24 = get_bsk24()  
BSk24_MASS_TABLE = get_bsk24_mass_table()
BSk24_EXPERIMENTAL_MASS_TABLE = get_bsk24_experimental_mass_table()
BSk24_VARIANS = get_bsk24_varians(full_data=True)
BSk24_VARIANS_MASS_TABLE = get_bsk24_varians_mass_table(full_data=True)
BSk24_VARIANS_EXT = get_bsk24_varians_ext()
BSk24_VARIANS_EXT_MASS_TABLE = get_bsk24_varians_ext_mass_table()

BSk24_dataset = {'BSk24_param' : 'bsk24.csv',
                 'BSk24_mass_table' : 'bsk24_mass_table.csv',
                 'BSk24_mass_table_exp' : 'bsk24_mass_table_exp.csv',
                 'BSk24_varians_param' : 'bsk24_varians.parquet',
                 'BSk24_varians_mass_table' : 'bsk24_varians_mass_table.parquet',
                 'BSk24_varians_sample_param' : 'bsk24_varians_sample.csv',
                 'BSk24_varians_sample_mass_table' : 'bsk24_varians_sample_mass_table.csv',
                 'BSk24_varians_ext_param' : 'bsk24_varians_ext.parquet',
                 'BSk24_varians_ext_mass_table' : 'bsk24_varians_ext_mass_table.parquet',
                 'BSk24_varians_ext_sample_param' : 'bsk24_varians_ext_sample.parquet',
                 'BSk24_varians_ext_sample_mass_table' : 'bsk24_varians_ext_mass_table.parquet'
                }

def load_df(name):
    file = BSk24_dataset[name]
    extension = Path(file).suffix
    
    if extension == 'parquet':
        df = read_parquet(file)
    elif extension == 'csv':
        df = read_csv(file)
    else:
        raise AttributeError(f"Unknown dataframe {name}")

# # Override __getattr__ to lazily load DataFrames
# def __getattr__(name):
#     return load_df(name)