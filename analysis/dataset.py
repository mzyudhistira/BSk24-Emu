import numpy as np
import pandas as pd

from input.load import load_df


def relative_skyrme(result_row):
    """
    Calculate the relative skyrme parameter wrt BSk24.

    Args:
        result_row (pd row): Row of the skyrme param df

    Returns:
        normalized_skyrme (list): Values of normalized skyrme param

    """
    bsk_param, _ = load_df("reg", "exp")
    normalized_skyrme = []

    for i in range(21):
        normalized_val = result_row[f"param({i+1:02d})"] - bsk_param.iloc[0, i]

        normalized_skyrme.append(normalized_val)

    return normalized_skyrme


def extract_result(result_row):
    """
    Extract dataset moment from a mass table

    Args:
        result_row (pd row): Row of the summary dataframe

    Returns:
        results (dict): Dictionary of RMS and dataset moment

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
