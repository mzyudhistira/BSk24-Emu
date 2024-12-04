import pandas as pd
import numpy as np
import ast

"""
Format data to pandas readable format
"""
def format_bsk24_varians(file):
    # Get the parameters tab
    params = pd.read_csv(file, nrows=43, sep=r"\s+", header=None).T
    
    header_row = params.head(1).values.tolist()[0]
    params.columns = header_row
    params = params.drop(0)
    params['sampling'].astype(int)

    # Get the mass tables
    mass_dict = []
    i = 1

    for col in mass_table.loc[:, 'sample_1':'sample_32119']:
        sample_mass_table = mass_table[['Z', 'N', 'A', f'sample_{i}']]
        sample_mass_table = sample_mass_table.rename(columns={f'sample_{i}':'mass'})
        formatted_dict = sample_mass_table.to_dict()

        mass_dict.append(formatted_dict)
        i += 1

    # Get the rms result
    rms_data = pd.read_csv(file, skiprows=774, sep=r"\s+", header=None).T
    header_row = rms_data.head(1).values.tolist()[0]
    rms_data.columns = header_row
    rms_data = rms_data.drop(0)

    # Combine it to a single df
    combined_data = pd.concat([params, rms_data], axis=1)
    combined_data['mass_table'] = mass_dict

    # Export to pandas readable format
    combined_data.to_csv('input/formatted_bsk24_varians.dat', sep=';', index=False)

    return


"""
Read the data
"""
def get_bsk24_data():
    df = pd.read_csv('input/literature_data/hfb24-dat', sep=r"\s+")

    return df

def get_bsk24_varians():
    df = pd.read_csv('input/literature_data/formatted_bsk24_varians.dat', sep=';')

    return df

def get_bsk24_varians_sample():
    df = pd.read_csv('input/literature_data/sample_bsk24_varians.dat', sep=';')

    return df

def get_bsk24_varian_mass_table(varian):
    mass_dict = ast.literal_eval(varian['mass_table'])
    mass_table = pd.DataFrame(mass_dict)

    return mass_table

def random_sampling(data, sample_number):
    np.random.shuffle(data)
    sample = data[:sample_number]

    return sample