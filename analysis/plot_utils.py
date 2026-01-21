import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator


def latex_figure(fraction=1.0, ratio=(np.sqrt(5) - 1, 2)):
    """Initialize figure size to suit the latex document

    Args:
        fraction (float): Fraction of the page, by default it stretch to fill the width of the page.
        ratio (tuple[float,float]): Ratio of the width and height of the figure, set to golden ratio by default.

    Returns:
        width, height: width and height to use in matplotlib
    """
    width_pt = 341.43306  # Width of the LaTeX document
    width_in = (width_pt / 72.27) * fraction
    w, h = ratio
    height_in = width_in * (h / w)

    return (width_in, height_in)


def set_tick(ax):
    """Set the tick of any plot to have 4 minor ticks inside major ticks

    Args:
        ax : matplotlib axis
    """

    if not isinstance(ax, (list, tuple, np.ndarray)):
        axes = [ax]
    else:
        axes = np.ravel(ax)

    for a in axes:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.tick_params(which="both", direction="in", width=1)
        a.tick_params(which="major", length=4)
        a.tick_params(which="minor", length=2, color="black")


def savefig(fig, ax, path):
    """Save the figure with some predefined settings

    Args:
        fig (plt.figure): Matplotlib figure object
        ax (plt.axis): Matplotlib axis object
        path (str): Path to save the figure
    """
    set_tick(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
