import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


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


def set_legend(ax):
    """Set legend and label of a plot in the nuclear landscape

    Args:
        ax : matplotlib axis
    """
    ax.legend(loc="lower right", borderpad=1, borderaxespad=1.5)
    ax.set_ylabel("Z")
    ax.set_xlabel("N")
