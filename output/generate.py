import numpy as np
import pandas as pd
from keras import models

# Internal Libraries
from config import *

output_data_dir = DATA_DIR / "output"


def generate_mass_table(model, complete_data, name):
    """
    Generate a mass table file
    """
    # Testing the model
    y = np.array(model.predict(complete_data[:, :-1])).flatten()
    print(y)

    # Generating data
    Z = np.array(complete_data[:, 1])
    N = np.array(complete_data[:, 0])
    BSk24 = np.array(complete_data[:, -1])
    diff = BSk24 - y

    # Saving the data
    data = {"Z": Z, "N": N, "BSk24": BSk24, "Prediction": y, "Difference": diff}

    df = pd.DataFrame(data)
    df = df.sort_values(by=["Z", "N"])
    df.to_csv(output_data_dir / f"{name}.dat", sep=";", index=False)

    return df


def mass_table(model, Z, N, input_data, target, name):
    """
    Generate a minimal mass table using a trained machine learning model
    """
    # Predict the target data
    y = np.array(model.predict(input_data)).flatten()

    # Generate the table to be saved
    data = {"Z": Z, "N": N, "Target": Target, "Prediction": y, "Difference": target - y}
    df = pd.DataFrame(data)
    df = df.sort_values(by=["Z", "N"])

    df.to_csv(output_data_dir / f"{name}.csv", sep=";", index=False)

    return df
