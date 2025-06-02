from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def plot_loss(loss, val_loss) -> None:
    """
    Plot loss and validation loss of the training

    Args:
        loss (str/Path): Path of the loss file
        val_loss (str/Path): Path of the validation loss file
    """
    loss = Path(loss)
    val_loss = Path(val_loss)

    loss_data = np.log10(np.loadtxt(loss))
    val_loss_data = np.log10(np.loadtxt(val_loss))
    save_dir = loss.parent

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(loss_data, color="blue")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(r"$\text{log}_{10}\text{loss}$")
    set_tick(ax[0])

    ax[1].plot(val_loss_data, color="green")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(r"$\text{log}_{10}\text{val_loss}$")
    set_tick(ax[1])

    fig.subplots_adjust(wspace=0.5)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/loss_plot.png")


def plot_rms(x, y, x_label) -> None:
    """
    Make plot of RMS Deviation

    Args:
        x (np arr): X axis
        y (np arr): Y axis (RMS Deviation)
        x_label (str): label of x
    """
    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMS Deviation (MeV)")
    ax.set_ylim(bottom=0)
    set_tick(ax)


def nuclear_landscape(N, Z, val, title="", colourbar_label="") -> None:
    """
    Make plot on the nuclear landscape

    Args:
        N (np arr): Neutron
        Z (np arr): Proton
        val (np arr): Plotted value
        title (str): Title of the plot, default to blank
        colourbar_label (str): Label of the colourbar, default to blank
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    scatter = ax.scatter(N, Z, c=val, s=4, cmap="inferno")
    ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_ylabel("Z")

    colour_bar = fig.colorbar(scatter, ax=ax)
    colour_bar.set_label(colourbar_label)

    set_tick(ax)


def set_tick(ax):
    """Set the tick of any plot to have 4 minor ticks inside major ticks

    Args:
        ax : matplotlib axis
    """
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="in", width=1)
    ax.tick_params(which="major", length=4)
    ax.tick_params(which="minor", length=2, color="black")
