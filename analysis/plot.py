from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def plot_loss(loss, val_loss):
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


def nuclear_landscape(N, Z, val, title="", colourbar_label=""):
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
