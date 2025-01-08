from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

import input
import model
import output
import training
import output
import utils
from config import *

def get_me_diff_over_std(series):
    return series.mean() / series.std()

def melt_mass_table(df, func = get_me_diff_over_std):    
    """Melt a mass table by grouping varians of the same isotope together

    Args:
        df (_type_): Dataframe of the mass table
        func (_type_, optional): Function used to melt. Defaults to get_me_diff_over_std.

    Returns:
        _type_: Mass table with grouped varians
    """
    return df.groupby(['Z', 'N']).agg({'Difference':func}).rename(columns={'Difference':'me_diff_over_std'}).reset_index()

def plot_landscape(df, ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    v_min, v_max = -4, 4

    norm = Normalize(vmin=v_min, vmax=v_max)
    scatter = ax.scatter(df['N'], df['Z'], c=np.clip(df['me_diff_over_std'], v_min, v_max), cmap='plasma', s=5)

    # fig.colorbar(scatter, ax=ax, label='Mass Excess Difference (MeV)/std')
    ax.set_xlabel('N')
    ax.set_ylabel('Z')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mass Excess Difference (MeV)/std')  # Label for the colorbar
    cbar.set_ticks(np.arange(v_min, v_max + 1, 1))

    return fig, ax, scatter

def plot_loss(mass_table_file):
    result_name = mass_table_file[:-4]
    batches = [32,16,4]
    epochs = [250,100,50]
    loss_dir = TRAINING_DATA_DIR / 'loss'
    loss_file = [loss_dir/f'{result_name}.batch={batches[i]}.epoch={epochs[i]}.stage{i+1}.loss.dat' for i in range(3)]
    val_loss_file = [loss_dir/f'{result_name}.batch={batches[i]}.epoch={epochs[i]}.stage{i+1}.val_loss.dat' for i in range(3)]

    loss_data = [np.loadtxt(file) for file in loss_file]
    val_loss_data = [np.loadtxt(file) for file in val_loss_file]

    loss_arr = [item for sublist in loss_data for item in sublist]
    val_loss_arr = [item for sublist in val_loss_data for item in sublist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(np.log(loss_arr), label='loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('log_10(Loss)')

    axes[1].plot(np.log(val_loss_arr), label='val_loss', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('log_10(Val_Loss)')

    return fig

def animate(i, dfs, scatter, ax):
    # Remove the old scatter plot
    for artist in ax.artists:
        artist.remove()
    for line in ax.lines:
        line.remove()

    # Plot the current DataFrame using plot_landscape
    scatter, ax = plot_landscape(dfs[i], ax)

    ax.set_title(f'Frame {i + 1}')  # Optional: title to show current frame index
    return scatter,  # Return the scatter for FuncAnimation

def create_animation(dfs):
    # Create the initial plot
    fig, ax = plt.subplots()
    scatter, ax = plot_landscape(dfs[0], ax)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(dfs), fargs=(dfs, scatter, ax), interval=1000, blit=False
    )

    # Display the animation
    plt.show()

    return ani

def analyse(mass_table_file):
    mass_table = pd.read_csv(DATA_DIR / 'output' / mass_table_file, sep=';')
    melted_mass_table = melt_mass_table(mass_table)
    
    rms_deviation = np.sqrt((mass_table['Difference']**2).mean())
    std_difference = mass_table['Difference'].std()

    landscape_plot, _, _ = plot_landscape(melted_mass_table)
    loss_plot = plot_loss(mass_table_file)
    plt.close(landscape_plot)
    plt.close(loss_plot)
 
    return {'mass_table' : mass_table, 'melted_mass_table' : melted_mass_table,
            'rms_dev' : rms_deviation, 'std_diff' : std_difference,
            'landscape_plot' : landscape_plot, 'loss_plot' : loss_plot}

def main():
    return