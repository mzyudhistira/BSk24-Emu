import numpy as np
import pandas as pd

from output import postprocess


def generate_mass_table(feature, target, prediction, path) -> None:
    """
    Generate a mass table csv file.

    Args:
        feature (np arr) : Feature of the dataset
        target (np arr) : Target dataset
        prediction (np arr) : ML Prediction
        path (str) : Path for saving the csv file

    """

    N = feature[:, 0]
    Z = feature[:, 1]
    diff = target - prediction

    df = pd.DataFrame(
        {
            "Z": np.rint(Z).astype(int),
            "N": np.rint(N).astype(int),
            "target": target,
            "prediction": prediction,
            "difference": diff,
        }
    )

    df = df.sort_values(by=["Z", "N"])
    df.to_csv(path, index=False)
