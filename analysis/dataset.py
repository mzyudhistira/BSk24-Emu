import numpy as np
import pandas as pd


def extract_result(result_row):
    """
    Extract dataset moment from a mass table

    Args:
        result_row (pd row): Row of the summary dataframe

    Returns:
        results (dic): Dictionary of RMS and dataset moment

    """
    mass_table = pd.read_csv(
        result_row["output_file"], usecols=["target", "prediction"]
    )

    moment = lambda column, name: {
        f"{name}_rms": rms(column),
        f"{name}_mean": column.mean(),
        f"{name}_std": column.std(),
        f"{name}_skew": column.skew(),
    }

    return (
        {"rms_dev": rms(mass_table["prediction"] - mass_table["target"])}
        | moment(mass_table["target"], "variant")
        | moment(mass_table["prediction"], "ml")
    )


def rms(array):
    """
    Calculate RMSE

    Args:
        array (np arr): Array to calculate

    Returns:
        rms (float)

    """
    return np.sqrt((array**2).mean())
