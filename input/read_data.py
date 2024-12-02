import pandas as pd

# Format data to pandas readable format
def get_bsk24_data():
    df = pd.read_csv('literature_data/hfb24-dat', sep=r"\s+")

    return df

def get_bsk24_varians():
    df = pd.read_csv('literature_data/formatted_BSk24_variations.dat')

    return df