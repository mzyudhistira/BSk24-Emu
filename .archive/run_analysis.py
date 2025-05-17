from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
from config import *


def plot_loss(mass_table_file):
    result_name = mass_table_file[:-4]
    batches = [32, 16, 4]
    epochs = [1500, 300, 50]
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(np.log(loss_arr), label="loss", color="blue")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("log_10(Loss)")

    axes[0, 1].plot(np.log(val_loss_arr), label="val_loss", color="green")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("log_10(Val_Loss)")

    axes[1, 0].plot(np.diff(np.log(loss_arr)), label="loss", color="blue")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("d log_10(Loss)")

    axes[1, 1].plot(np.diff(np.log(val_loss_arr)), label="val_loss", color="green")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("d log_10(Val_Loss)")

    # axes[2, 0].plot(np.diff(np.log(loss_arr), n=2), label="loss", color="blue")
    # axes[2, 0].set_xlabel("Epoch")
    # axes[2, 0].set_ylabel("d2 log_10(Loss)")
    #
    # axes[2, 1].plot(np.diff(np.log(val_loss_arr), n=2), label="val_loss", color="green")
    # axes[2, 1].set_xlabel("Epoch")
    # axes[2, 1].set_ylabel("d2 log_10(Val_Loss)")

    # return fig


def plot_uncertainty(mass_table, ax=None, bin_edges=[0, 1, 2, 3]):
    if ax is None:
        fig, ax = plt.subplots()

    mass_table_unc = mass_table.copy()
    mass_table_unc["m_std"] = np.floor(mass_table["m_std"]).clip(0, 3)
    colour = {0: "#FCD93D", 1: "#01A5EA", 2: "#008110", 3: "#FF0703"}
    label = {
        0: r"$\sigma \leq 1$ MeV",
        1: r"$1 < \sigma \leq 2$ MeV",
        2: r"$2 < \sigma \leq 3$ MeV",
        3: r"$\sigma > 3$ MeV",
    }
    plotted_values = [mass_table_unc[mass_table_unc["m_std"] == i] for i in bin_edges]

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


def plot_uncertainty_all(mass_table, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    scatter = ax.scatter(
        mass_table["N"], mass_table["Z"], c=mass_table["m_std"], s=4, cmap="inferno"
    )
    ax.set_title(f"Uncertainty of Mass Prediction Across the Nuclear Chart")
    ax.set_xlabel("N")
    ax.set_ylabel("Z")

    colour_bar = fig.colorbar(scatter, ax=ax)
    colour_bar.set_label(r"$\sigma$ (MeV)")

    utils.plot.set_tick(ax)

    return fig, ax


def plot_histogram_nucleus(N, Z, mass_table, resolution=0.01, ax=None):
    """Plot the mass distribution of a given nucleus

    Args:
        N (int): Neutron number
        Z (int): Proton number
        mass_table (dataframe): Mass table from the output
        resolution (float, optional): Smallest scale of the dataset. Defaults to 0.01 MeV.
        ax (plt.ax, optional): Pre-defined axis, used if wants to put the plot on top of another. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()

    nucleus_mass = mass_table[(mass_table["N"] == N) & (mass_table["Z"] == Z)]
    # mass_range = nucleus_mass["Prediction"].nanmax() - nucleus_mass["Prediction"].nanmin()

    mass_range = np.nanmax(nucleus_mass["Prediction"].values) - np.nanmin(
        nucleus_mass["Prediction"].values
    )

    bins = int(np.ceil(mass_range / resolution))

    ax.hist(nucleus_mass["Prediction"], bins=bins, label="ML")
    ax.set_title(f"Mass Distribution of Nucleus N={N}, Z={Z}")
    ax.set_xlabel("Mass (MeV)")
    ax.set_ylabel("Frequency")

    fig.show()


def plot_deviation(all_mass_table, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    mass_table_dev = all_mass_table.groupby(["Z", "N"]).agg(
        {"Difference": "mean", "BSk24": "std"}
    )
    mass_table_dev.columns = ["mean_diff", "std_bsk24"]
    mass_table_dev = mass_table_dev.reset_index()

    mass_table_dev["Deviation"] = (
        mass_table_dev["mean_diff"] / mass_table_dev["std_bsk24"]
    )

    mass_table_dev["Deviation"] = np.floor(mass_table_dev["Deviation"]).clip(0, 3)

    colour = {0: "#FCD93D", 1: "#01A5EA", 2: "#008110", 3: "#FF0703"}
    label = {
        0: r"$\epsilon = \sigma$",
        1: r"$\epsilon = 2 * \sigma$",
        2: r"$\epsilon = 3 * \sigma$",
        3: r"$\epsilon > 3 * \sigma$",
    }
    plotted_values = [
        mass_table_dev[mass_table_dev["Deviation"] == i] for i in range(4)
    ]

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


def analyse_single_variant(mass_table_file):
    return


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

    # Extract data
    rms_deviation = np.sqrt((all_mass_table["Difference"] ** 2).mean())
    std_difference = all_mass_table["Difference"].std()

    print(f"rms_deviation: {rms_deviation}")
    print(f"std_difference: {std_difference}")
    print(f'avg: {all_mass_table["Difference"].mean()}')

    # Plot
    # f_unc, ax_unc = plot_uncertainty(mass_table)
    # f_unca, ax_unca = plot_uncertainty_all(mass_table)
    # f_dev, ax_dev = plot_deviation(all_mass_table)
    # f_loss, ax_loss = plot_loss(file)


if __name__ == "__main__":
    main()
