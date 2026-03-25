from pathlib import Path

import numpy as np
import pandas as pd

from input.load import load_df


def opt_data_test() -> pd.DataFrame:
    """Load dataset of optimum training test

    Returns:
        dataset (pd.DataFrame): Summary of the dataset
    """
    dataset: pd.DataFrame = pd.read_csv("data/summary/optimum_data_test.csv")

    # Extract parameter
    dataset["variant_id"] = [
        var_str.split("_")[1] for var_str in dataset["run_name"].values
    ]
    dataset["percent_train_data"] = [
        round(float(var_str.split("_")[2]), 2) * 80
        for var_str in dataset["run_name"].values
    ]

    return dataset


def extract_variant_moment(variant_mt) -> pd.Series:
    """Calculate the dataset moment of each variant
    Args:
        variant_mt (pd.DataFrame): DataFrame of a variant to be analysed.

    Returns:
        moment_df (pd.Series): Series of variant's moments, including some comparative quantities.
    """
    target = variant_mt["target"]
    prediction = variant_mt["prediction"]

    rms_dev = rms(target - prediction)

    moment_df = pd.Series(
        {
            "rms_dev": rms_dev,
            "drms": rms(target) - rms(prediction),
            "r_std": prediction.std() / target.std(),
            "variant_rms": rms(target),
            "variant_mean": target.mean(),
            "variant_std": target.std(),
            "variant_skew": target.skew(),
            "ml_rms": rms(prediction),
            "ml_mean": prediction.mean(),
            "ml_std": prediction.std(),
            "ml_skew": prediction.skew(),
        }
    )

    return moment_df


def goriely_uncertainty(data_type="forward") -> pd.DataFrame:
    """Load the dataset from Goriely et al's study, which can be used to make uncertainty plot like Fig.4 on their paper.
    Args:
        data_type (str): Type of the dataset from the study. Possible options are backward and forward. From the BFMC in the paper.

    Returns:
        unc_data (pd.DataFrame) = Uncertainty dataset
    """

    if data_type == "backward":
        data = pd.read_parquet("data/input/bsk24_variants_mass_table.parquet")
    elif data_type == "forward":
        data = pd.read_parquet("data/input/bsk24_variants_ext_mass_table.parquet")
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    unc_data = data.groupby(["Z", "N"]).agg(m_sigma=("m", "std")).reset_index()

    def categorize(values, labels):
        bins = [0, 1, 2, 3, float("inf")]
        return pd.cut(values, bins=bins, labels=labels, right=True)

    unc_data["sigma"] = categorize(
        unc_data["m_sigma"],
        [
            r"$\sigma \leq 1$ MeV",
            r"$1 < \sigma \leq 2$ MeV",
            r"$2 < \sigma \leq 3$ MeV",
            r"$\sigma \geq 4$ MeV",
        ],
    )

    return unc_data


def epsilon_sigma_dataset(train_data="full") -> pd.DataFrame:
    """Load sigma and epsilon values of nuclei.
    Args:
        train_data (str): The ML dataset to plot, train with certain percentage of input data. Possible options are 025, 05, 1, 2, 4, 8, and full (24). Default to full.

    Returns:
        grouped (pd.DataFrame) : Dataset
    """

    if train_data == "full":
        dataset = pd.read_parquet("data/result/full_mass_table.parquet")
    else:
        dataset = pd.read_parquet(f"data/result/full_mass_table_{train_data}.parquet")

    grouped = (
        dataset.groupby(["Z", "N"])
        .agg(
            target_mean=("target", "mean"),
            target_std=("target", "std"),
            prediction_mean=("prediction", "mean"),
            prediction_std=("prediction", "std"),
        )
        .reset_index()
    )

    def categorize(values, labels):
        bins = [0, 1, 2, 3, float("inf")]
        return pd.cut(values, bins=bins, labels=labels, right=True)

    raw_epsilon = (
        np.abs(grouped["prediction_mean"] - grouped["target_mean"])
        / grouped["target_std"]
    )

    grouped["epsilon"] = categorize(
        raw_epsilon,
        [
            r"$\epsilon \leq 1$ MeV",
            r"$1 < \epsilon \leq 2$ MeV",
            r"$2 < \epsilon \leq 3$ MeV",
            r"$\epsilon \geq 4$ MeV",
        ],
    )

    grouped["sigma"] = categorize(
        grouped["prediction_std"],
        [
            r"$\sigma \leq 1$ MeV",
            r"$1 < \sigma \leq 2$ MeV",
            r"$2 < \sigma \leq 3$ MeV",
            r"$\sigma \geq 4$ MeV",
        ],
    )

    grouped["sigma_t"] = categorize(
        grouped["target_std"],
        [
            r"$\sigma \leq 1$ MeV",
            r"$1 < \sigma \leq 2$ MeV",
            r"$2 < \sigma \leq 3$ MeV",
            r"$\sigma \geq 4$ MeV",
        ],
    )

    return grouped


def extract_moment_old_dataset() -> pd.DataFrame:
    """Extract the moment and relevant properties of the previous ML model, where variants parameters were incorporated

    Returns:
        data (pd.DataFrame) : Old dataset

    """
    data_dir: Path = Path("data/old")

    files: list[Path] = [
        f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".csv"
    ]

    data_list = []
    for file in files[:]:
        run_data: pd.DataFrame = pd.read_csv(file, sep=";")

        run_properties = {}
        run_properties["dataset_type"] = str(file).split("/")[-1].split("_")[0]
        run_properties["var_train"] = str(file).split("/")[-1].split("_")[1]
        run_properties["var_predict"] = (
            str(file).split("/")[-1].split("_")[2].split(".")[0]
        )
        run_properties["rms_dev"] = rms(run_data["Difference"])
        run_properties["avg_diff"] = run_data["Difference"].mean()
        run_properties["std_diff"] = run_data["Difference"].mean()

        data_list.append(run_properties)

    data = pd.DataFrame(data_list)
    data.to_csv("data/result/old_run.csv", index=False)

    return data


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
        normalized_val = result_row[f"param({i + 1:02d})"] - bsk_param.iloc[0, i]

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
