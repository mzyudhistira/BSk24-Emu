from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from . import plot_utils
from . import dataset


def plot_me_bsk_comparison(path: str) -> None:
    """Plot the mass excess predictions of BSk1-32 for A=195 chain. This plot is used in introduction.

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure(ratio=(9, 5)))

    # Load all HFB mass tables
    files = [str(mt) for mt in Path("data/others/bsks_mt").iterdir()]
    labels = [file.split("-")[0].split("/")[-1] for file in files]
    labels.sort()
    mass_tables = [pd.read_csv(file, skiprows=[0, 2], sep="\\s+") for file in files]

    # Define  BSk32 Mass table and filter it
    bsk32_mt = mass_tables[-1]
    bsk32_mt["N"] = bsk32_mt["A"] - bsk32_mt["Z"]
    bsk32_mt = bsk32_mt[(bsk32_mt["A"] == 195) & (bsk32_mt["N"] >= 110)]

    # Plot experimental mass
    ax.scatter(
        bsk32_mt["N"].iloc[-10:],
        bsk32_mt["Mexp-Mcal"].iloc[-10:],
        label="Exp",
        zorder=10,
        color="black",
    )

    # Plot all mass excess relative to BSk-32
    for i in range(20, len(files[:-1])):
        ref = bsk32_mt[["N", "Mcal"]].rename(columns={"Mcal": "Mcal_ref"})

        mass_table = mass_tables[i]
        mass_table["N"] = mass_table["A"] - mass_table["Z"]
        mass_table = mass_table[(mass_table["A"] == 195) & (mass_table["N"] >= 110)]
        mass_table = mass_table.merge(ref, on="N", how="inner")

        ax.plot(
            mass_table["N"],
            mass_table["Mcal"] - mass_table["Mcal_ref"],
            # label=labels[i],
            zorder=5,
        )
    ax.plot([], [], color="black", label="BSk21-31")

    ax.set_xlabel("N")
    ax.set_ylabel(r"$m - m_{\mathrm{BSk32}}\;(\mathrm{MeV})$")
    ax.legend()

    plot_utils.savefig(fig, ax, path)


def plot_correlation_illustration(path: str) -> None:
    """Plot an illustration of Pearson and Spearman correlation coefficient, where Spearmann~1 and Pearson ~0

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure(ratio=(9, 5)))

    x = np.linspace(0.01, 0.99, 250)
    y = 0.5 + 0.1 * np.tan(np.pi * (x + 0.5)) + 3e-2 * np.random.randn(len(x))

    # correlations
    r_pearson = pearsonr(x, y)[0]
    r_spearman = spearmanr(x, y)[0]

    print(f"Pearson r  = {r_pearson:.3f}")
    print(f"Spearman ρ = {r_spearman:.3f}")

    ax.scatter(x, y, s=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plot_utils.savefig(fig, ax, path)


def plot_loss_convergence(path: str) -> None:
    """Plot the convergence of loss and val loss of the initial model

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(1, 2, figsize=plot_utils.latex_figure(ratio=(12, 4.5)))

    # Load the loss plot
    loss_data = np.loadtxt("data/result/2025-06-01 19:31_Dataset=100.0%/loss.dat")

    val_loss_data = np.loadtxt(
        "data/result/2025-06-01 19:31_Dataset=100.0%/val_loss.dat"
    )

    # Plot loss
    ax[0].plot(loss_data, color="blue", lw=0.6)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(r"$\text{loss (MeV}^2)$")
    ax[0].set_yscale("log")

    # Plot val loss
    ax[1].plot(val_loss_data, color="green", lw=0.6)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(r"$\text{val loss (MeV}^2)$")
    ax[1].set_yscale("log")

    plot_utils.savefig(fig, ax, path)


def plot_robustness_samebase(path: str) -> None:
    """Plot the robustness test of the model with the same base configuration. Left plot shows the impact of additional neuron, while right plot shows the impact of dropout architecture

    Args:
        path (str): path to save the figure
    """

    fig, ax = plt.subplots(1, 2, figsize=plot_utils.latex_figure(ratio=(18, 7)))

    # Load data
    data_neuron_addition = pd.read_csv("summary.csv", sep=",").iloc[42:48, :]
    data_neuron_drop = pd.read_csv("summary.csv", sep=",").iloc[24:30, :]

    ax[0].scatter(
        data_neuron_addition["note"].astype(int), data_neuron_addition["rms_dev"]
    )
    ax[0].set_ylabel("RMS Deviation (MeV)")
    ax[0].set_xlabel("Number of additional layer(s)")
    ax[0].set_ylim(0, 1.75)
    ax[0].set_xticks(np.arange(0, 6))

    ax[1].scatter(
        data_neuron_drop["note"].astype(float) * 100, data_neuron_drop["rms_dev"]
    )
    ax[1].set_xlabel("Dropout rate (\\%)")
    ax[1].set_ylim(0)
    ax[1].set_xticks(np.arange(20, 90, 20))

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_diffbase(path: str) -> None:
    """Plot the robustness test of the model with different base configuration. Left plot shows inverse sequential architecture, while right plot shows bottleneck architecture.

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(1, 2, figsize=plot_utils.latex_figure(ratio=(18, 7)))

    # Load data
    data_inverse_sequential = pd.read_csv("summary.csv", sep=",").iloc[7:17]
    data_bottleneck = pd.read_csv("summary.csv", sep=",").iloc[17:24]

    # Plot inverse sequential result
    ax[0].scatter(np.arange(1, 11), data_inverse_sequential["rms_dev"])
    ax[0].set_ylabel("RMS Deviation (MeV)")
    ax[0].set_xticks(np.arange(1, 11, 2))
    ax[0].set_ylim(bottom=0, top=0.8)

    # Plot bottleneck result
    ax[1].scatter(np.arange(1, 8), data_bottleneck["rms_dev"])
    ax[1].set_xticks(np.arange(1, 8, 2))
    ax[1].set_ylim(bottom=0, top=1.25)

    fig.supxlabel("Multiplication Factor")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stability_one(path: str) -> None:
    """Plot the stability test of the ML model on several variants
    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(
        1,
        3,
        figsize=plot_utils.latex_figure(ratio=(18, 7)),
        sharey=True,
    )

    # Load data
    stability_data: pd.DataFrame = pd.read_csv("summary.csv", sep=",").iloc[137:-2]

    variant_test: list[int] = [1, 13, 1420]

    for i in range(3):
        v_test: pd.DataFrame = stability_data[
            stability_data["run_name"].str.startswith(f"Variant_{variant_test[i]}_")
        ]
        rms_dev: pd.Series = v_test["rms_dev"]

        ax[i].hist(rms_dev)
        ax[i].set_title(f"Variant {variant_test[i]}")

    fig.supylabel("Frequency")
    fig.supxlabel("RMS Deviation (MeV)")

    plot_utils.savefig(fig, ax, path)


def plot_percent_train(path: str) -> None:
    """Plot the change of RMSE due to different amount of training data

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure(ratio=(9, 5)))

    # Load data
    opt_data_test = dataset.opt_data_test()

    sns.boxplot(
        x="percent_train_data",
        y="rms_dev",
        data=opt_data_test,
        fliersize=3,
        width=0.5,
        ax=ax,
    )

    # Adjust ticks to show only half of them
    ticks = ax.get_xticks()
    labels: list[str] = [item.get_text()[:4] for item in ax.get_xticklabels()]

    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2])

    ax.set_ylabel("RMS Deviation (MeV)")
    ax.set_xlabel(r"Train Data (\%)")
    ax.set_ylim(bottom=0)

    plot_utils.savefig(fig, ax, path)


def plot_ic50_analysis(path: str) -> None:
    """Plot IC50 analysis to determine the minimum amount of training data

    Args:
        path (str): path to save the figure
    """

    fig, ax = plt.subplots(1, 2, figsize=plot_utils.latex_figure(ratio=(18, 7)))

    # Groupby
    compact_df = dataset.opt_data_test()
    compact_df = compact_df.groupby("percent_train_data", as_index=False).agg(
        mean=("rms_dev", "mean"),
        iqr=("rms_dev", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    )

    # Determine the optimum
    ax[0].scatter(compact_df["percent_train_data"], compact_df["mean"])
    ax[0].set_ylabel("RMS Deviation (MeV)")

    ax[1].scatter(compact_df["percent_train_data"], compact_df["iqr"])
    ax[1].set_ylabel("IQR (MeV)")

    # IC50-like determination
    def det_ic50(col):
        val_max = col.max()
        val_min = col.min()

        half_pos = (val_max - val_min) / 2 + val_min
        delta = col.apply(lambda x: x - half_pos)
        lower_than_half = delta[delta < 0]
        higher_than_half = delta[delta > 0]
        first_closest_value = [
            np.abs(lower_than_half.iloc[0]),
            np.abs(higher_than_half.iloc[-1]),
        ]

        return delta.index[delta.abs() == min(first_closest_value)]

    ic50_mean = compact_df.loc[det_ic50(compact_df["mean"])][
        "percent_train_data"
    ].values[0]

    ic50_iqr = compact_df.loc[det_ic50(compact_df["iqr"])]["percent_train_data"].values[
        0
    ]
    mean_thresh = compact_df[compact_df["percent_train_data"] == ic50_mean][
        "mean"
    ].values[0]
    iqr_thresh = compact_df[compact_df["percent_train_data"] == ic50_iqr]["iqr"].values[
        0
    ]

    ax[0].plot(
        np.linspace(0, 100, 100),
        mean_thresh * np.ones(100),
        color="red",
    )
    ax[1].plot(
        np.linspace(0, 100, 100),
        iqr_thresh * np.ones(100),
        color="red",
    )

    print(f"Mean: {ic50_mean}")
    print(f"IQR: {ic50_iqr}")

    fig.supxlabel(r"Training Data (\%)")

    plot_utils.savefig(fig, ax, path)


def plot_rmse_dist(path: str) -> None:
    """Plot the RMSE distribution of full scale simulation on 24% dataset

    Args:
        path (str): path to save the figure
    """

    fig, ax = plt.subplots(
        1, 1, figsize=plot_utils.latex_figure(fraction=0.8, ratio=(3, 2))
    )

    # Load dataset
    data: pd.DataFrame = pd.read_csv("data/summary/full_scale.csv")

    ax.hist(data["rms_dev"])
    ax.set_yticks(np.arange(0, 9000, 2000))
    ax.set_xlabel(r"$\text{RMSE}_\text{v, ML}$")
    ax.set_ylabel("Frequency")

    plot_utils.savefig(fig, ax, path)


def plot_moment_correlation(path: str) -> None:
    """Plot correlation of variant's and ML's moments

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(
        1, 1, figsize=plot_utils.latex_figure(ratio=(4, 3), fraction=0.8)
    )

    # Load the dataset
    full_result = pd.read_parquet("data/result/full_mass_table.parquet")
    moment_df = (
        full_result.groupby(["variant_id"])
        .apply(dataset.extract_variant_moment)
        .reset_index()
    )
    del full_result

    # Correlation test
    target_columns = ["variant_rms", "variant_mean", "variant_std", "variant_skew"]
    result_columns = [
        "rms_dev",
        "r_std",
        "drms",
        "ml_rms",
        "ml_mean",
        "ml_std",
        "ml_skew",
    ]

    corr_matrix = pd.DataFrame(
        index=target_columns, columns=result_columns, dtype=float
    )

    for index in target_columns:
        for column in result_columns:
            corr = moment_df[[index, column]].corr(method="spearman").iloc[0, 1]
            corr_matrix.loc[index, column] = round(corr, 3)

    ticks = [-1, -0.5, 0, 0.5, 1]

    # Make coloured figure
    ax = sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"ticks": ticks},
    )

    ax.collections[0].colorbar.set_ticklabels(ticks)

    ax.set_xticklabels(
        [
            r"RMSE$_\text{v,ML}$",
            r"$r_\sigma$",
            r"$\Delta$RMS",
            r"$\mu_\text{RMS}^\text{ML}$",
            r"$\mu^\text{ML}$",
            r"$\sigma^\text{ML}$",
            r"$\gamma^\text{ML}$",
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticklabels(
        [r"$\mu_\text{RMS}^v$", r"$\mu^v$", r"$\sigma^v$", r"$\gamma^v$"], rotation=0
    )

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_mu(path) -> None:
    """Plot the Delta mu_v,ML across the nuclear chart

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=plot_utils.latex_figure(ratio=(16, 9)))

    # Load data
    data: pd.DataFrame = pd.read_parquet("data/result/full_mass_table.parquet")
    aggregated_data: pd.DataFrame = (
        data.groupby(by=["Z", "N"])
        .agg(
            target_mean=("target", "mean"),
            target_std=("target", "std"),
            prediction_mean=("prediction", "mean"),
            prediction_std=("prediction", "std"),
        )
        .reset_index()
    )
    del data

    aggregated_data["dmu"] = (
        aggregated_data["prediction_mean"] - aggregated_data["target_mean"]
    )

    dmu_plot = ax.scatter(
        aggregated_data["N"],
        aggregated_data["Z"],
        c=aggregated_data["dmu"],
        s=0.5,
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    plt.colorbar(dmu_plot, ax=ax, label=r"$\Delta \bar{\mu}_\text{v,ML}$")

    plot_utils.savefig(fig, ax, path)


def plot_rstd(path) -> None:
    """Plot the Delta r_sigma across the nuclear chart

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=plot_utils.latex_figure(ratio=(16, 9)))

    # Load data
    data: pd.DataFrame = pd.read_parquet("data/result/full_mass_table.parquet")
    aggregated_data: pd.DataFrame = (
        data.groupby(by=["Z", "N"])
        .agg(
            target_mean=("target", "mean"),
            target_std=("target", "std"),
            prediction_mean=("prediction", "mean"),
            prediction_std=("prediction", "std"),
        )
        .reset_index()
    )
    del data

    aggregated_data["rsigma"] = (
        aggregated_data["prediction_std"] / aggregated_data["target_std"]
    )

    rsigma_plot = ax.scatter(
        aggregated_data["N"],
        aggregated_data["Z"],
        c=aggregated_data["rsigma"],
        s=0.5,
        cmap="coolwarm",
        vmin=0,
        vmax=2,
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    plt.colorbar(rsigma_plot, ax=ax, label=r"$r_\sigma$")

    plot_utils.savefig(fig, ax, path)


def plot_gap_n_asymmetry():
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    data = dataset.build_gap_dataset()

    asymmetry_plot = ax[0].scatter(data["N"], data["Z"], c=data["as"], s=3)
    fig.colorbar(asymmetry_plot, ax=ax[0], label="Asymmetry")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("Z")

    as_list = [i for i in range(1, 60)]
    iqr_list = []

    for i in as_list:
        test_data = data[data["as"] <= i]
        iqr_list.append(
            test_data["eps_delta"].quantile(0.75)
            - test_data["eps_delta"].quantile(0.25)
        )

    ax[1].plot(as_list, iqr_list)
    ax[1].set_xlabel("Asymmetry")
    ax[1].set_ylabel(r"$\epsilon_\Delta$ (\%)")

    plt.tight_layout()
    fig.savefig("../5_Writing/chapters/4_full_scale/image/gap_n_asymmetry.png")
    plt.close()


def plot_sigma_propagation():
    data = dataset.build_grouped_dataset()

    fig, ax = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.5, 1.5]}
    )

    # Uncertainty propagation
    pivot = data.pivot(index="N", columns="Z", values="target_std")

    N_vals = pivot.index.values
    Z_vals = pivot.columns.values
    sigma = pivot.values

    dsigma_dN, dsigma_dZ = np.gradient(sigma, N_vals, Z_vals)
    grad_N = pd.DataFrame(
        dsigma_dN,
        index=pivot.index,
        columns=pivot.columns,
    )

    grad_Z = pd.DataFrame(
        dsigma_dZ,
        index=pivot.index,
        columns=pivot.columns,
    )

    data = data.merge(
        grad_N.stack().rename("dsigma_dn"),
        left_on=["N", "Z"],
        right_index=True,
    )

    data = data.merge(
        grad_Z.stack().rename("dsigma_dz"),
        left_on=["N", "Z"],
        right_index=True,
    )

    data["dsigma"] = np.sqrt(data["dsigma_dn"] ** 2 + data["dsigma_dz"] ** 2)

    plot = ax[0].scatter(data["N"], data["Z"], c=data["dsigma"], s=4, vmin=0, vmax=0.3)
    fig.colorbar(plot, ax=ax[0], label=r"$\Delta\sigma$")

    ax[0].set_xlabel("N")
    ax[0].set_ylabel("Z")

    # Gradiant distribution
    hist, bins = np.histogram(np.abs(data["dsigma"]), bins=np.linspace(0, 3.5, 15))
    iqr = []
    mean = []
    for i in range(1, len(bins)):
        data_i = data[
            (np.abs(data["eps"]) > bins[i - 1]) & (np.abs(data["eps"]) < bins[i])
        ]

        mean.append(data_i["eps"].mean())
        iqr.append(data_i["eps"].quantile(0.75) - data_i["eps"].quantile(0.25))

    ax[1].plot(bins[:-1], iqr[:])
    ax[1].set_xlabel(r"$|\epsilon|$")
    ax[1].set_ylabel(r"IQR")

    ax12 = ax[1].twinx()
    ax12.plot(bins[:-1], mean[:], color="#ff7f0e")
    ax12.set_ylabel(r"$\bar{\Delta\sigma}$")

    plt.tight_layout()
    plt.savefig("../5_Writing/chapters/5_uq_cost/image/unc_propagation.png")
    plt.close()


def plot_pure_uncertainty():
    data = dataset.build_grouped_dataset()
    plt.scatter(data["N"], data["Z"], c=data["prediction_std"])

    plt.tight_layout()
    plt.savefig("../5_Writing/chapters/5_uq_cost/image/unc_pure.png")
    plt.close()
