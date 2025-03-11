from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
from config import *


def plot_loss(mass_table_file):
    result_name = mass_table_file[:-4]
    batches = [32, 16, 4]
    epochs = [250, 100, 50]
    loss_dir = TRAINING_DATA_DIR / "loss"
    loss_file = [
        loss_dir
        / f"{result_name}.batch={batches[i]}.epoch={epochs[i]}.stage{i+1}.loss.dat"
        for i in range(3)
    ]
    val_loss_file = [
        loss_dir
        / f"{result_name}.batch={batches[i]}.epoch={epochs[i]}.stage{i+1}.val_loss.dat"
        for i in range(3)
    ]

    loss_data = [np.loadtxt(file) for file in loss_file]
    val_loss_data = [np.loadtxt(file) for file in val_loss_file]

    loss_arr = [item for sublist in loss_data for item in sublist]
    val_loss_arr = [item for sublist in val_loss_data for item in sublist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(np.log(loss_arr), label="loss", color="blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("log_10(Loss)")

    axes[1].plot(np.log(val_loss_arr), label="val_loss", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("log_10(Val_Loss)")

    return fig


def plot_uncertainty(mass_table, ax=None, bin_edges=[0, 1, 2, 3]):
    if ax is None:
        fig, ax = plt.subplots()

    mass_table["m_std"] = np.floor(mass_table["m_std"]).clip(0, 3)
    colour = {0: "#FCD93D", 1: "#01A5EA", 2: "#008110", 3: "#FF0703"}
    label = {
        0: r"$\sigma \leq 1$ MeV",
        1: r"$1 < \sigma \leq 2$ MeV",
        2: r"$2 < \sigma \leq 3$ MeV",
        3: r"$\sigma > 3$ MeV",
    }
    plotted_values = [mass_table[mass_table["m_std"] == i] for i in bin_edges]

    for i in range(4):
        scatter = ax.scatter(
            plotted_values[i]["N"],
            plotted_values[i]["Z"],
            color=colour[i],
            label=label[i],
            s=5,
        )

    utils.plot.set_legend(ax)
    utils.plot.set_legend(ax)

    return fig, ax


def plot_deviation(all_mass_table, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    mass_table = all_mass_table.groupby(["Z", "N"]).agg(
        {"Difference": "mean", "BSk24": "std"}
    )
    mass_table.columns = ["mean_diff", "std_bsk24"]
    mass_table = mass_table.reset_index()

    mass_table["Deviation"] = mass_table["mean_diff"] / mass_table["std_bsk24"]

    mass_table["Deviation"] = np.floor(mass_table["Deviation"]).clip(0, 3)

    colour = {0: "#FCD93D", 1: "#01A5EA", 2: "#008110", 3: "#FF0703"}
    label = {
        0: r"$\epsilon = \sigma_\text{BSk24v}$",
        1: r"$\epsilon = 2 \sigma_\text{BSk24v}$",
        2: r"$\epsilon = 3 \sigma_\text{BSk24v}$",
        3: r"$\epsilon > 3 \sigma_\text{BSk24v}$",
    }
    plotted_values = [mass_table[mass_table["Deviation"] == i] for i in range(4)]

    for i in range(4):
        scatter = ax.scatter(
            plotted_values[i]["N"],
            plotted_values[i]["Z"],
            color=colour[i],
            label=label[i],
            s=5,
        )

    utils.plot.set_legend(ax)
    utils.plot.set_tick(ax)

    return fig, ax


def main(file):
    # Get file
    mass_table_file = ""
    loss_file = ""
    val_loss_file = ""

    # Extract the Dataframe
    all_mass_table = pd.read_csv(file, sep=";")
    mass_table = all_mass_table.groupby(["Z", "N"]).agg({"Prediction": ["mean", "std"]})
    mass_table.columns = ["m_mean", "m_std"]
    mass_table = mass_table.reset_index()
    print(mass_table)

    # Extract data
    rms_deviation = np.sqrt((all_mass_table["Difference"] ** 2).mean())
    std_difference = all_mass_table["Difference"].std()

    print(rms_deviation, std_difference)

    # Plot
    f_unc, ax_unc = plot_uncertainty(mass_table)
    f_dev, ax_dev = plot_deviation(all_mass_table)
    # f_loss, ax_loss = plot_loss(file)

    return


def write_result():
    return


if __name__ == "__main__":
    main()
