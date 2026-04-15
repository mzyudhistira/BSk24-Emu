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
    plotted_bsk = [20, 23, 28, 29, 30]  # BSk number -1 due to python indexing
    for i in plotted_bsk:
        ref = bsk32_mt[["N", "Mcal"]].rename(columns={"Mcal": "Mcal_ref"})

        mass_table = mass_tables[i]
        mass_table["N"] = mass_table["A"] - mass_table["Z"]
        mass_table = mass_table[(mass_table["A"] == 195) & (mass_table["N"] >= 110)]
        mass_table = mass_table.merge(ref, on="N", how="inner")

        ax.plot(
            mass_table["N"],
            mass_table["Mcal"] - mass_table["Mcal_ref"],
            label=f"HFB-{labels[i][-2:]}",
            zorder=5,
        )

    ax.set_xlabel("N")
    ax.set_ylabel(r"$m - m_{\mathrm{BSk32}}\;(\mathrm{MeV})$")
    ax.legend(ncol=2, loc=3)

    plot_utils.savefig(fig, ax, path)


def plot_bsk_res(path: str) -> None:
    """Plot the mass and INM predictions made by BSk22-26

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure(fraction=0.8))

    data = pd.read_csv("data/others/BSk22-26_INM.csv")
    data = data.iloc[:, [0, 1, 3, 5, 7]].rename(
        columns={
            "rms_dev(M)": "$M$",
            "rms_dev(M_nr)": "$M_{nr}$",
            "rms_dev(S_n)": "$S_n$",
            "rms_dev(Q_beta)": "$Q_\\beta$",
        }
    )
    data = pd.melt(data, id_vars="model", var_name="property", value_name="value")

    sns.barplot(data=data, x="property", y="value", hue="model", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("RMS Deviation (MeV)")
    plt.legend(ncol=2, loc=4, fontsize="small")

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


def plot_epsilon(path) -> None:
    """Plot the epsilon across the nuclear chart

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=plot_utils.latex_figure(ratio=(16, 9)))

    # Load Data
    data = dataset.epsilon_sigma_dataset()

    colour_map = {
        r"$\epsilon \leq 1$ MeV": "#FBD900",
        r"$1 < \epsilon \leq 2$ MeV": "#00AAF0",
        r"$2 < \epsilon \leq 3$ MeV": "#098000",
        r"$\epsilon \geq 4$ MeV": "#FF1503",
    }

    hue_order = [
        r"$\epsilon \leq 1$ MeV",
        r"$1 < \epsilon \leq 2$ MeV",
        r"$2 < \epsilon \leq 3$ MeV",
        r"$\epsilon \geq 4$ MeV",
    ]
    sns.scatterplot(
        data=data,
        x="N",
        y="Z",
        hue="epsilon",
        palette=colour_map,
        s=1.8,
        hue_order=hue_order,
        edgecolor="none",
    )

    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    ax.legend(title="", loc="lower right", markerscale=3)

    plot_utils.savefig(fig, ax, path)


def plot_uncertainty(
    path: str, train_data: str = "full", variant: bool = False
) -> None:
    """Plot the uncertainty of ML dataset
    Args:
        path (str): Path to save the figure
        train_data (str): The ML dataset to plot, train with certain percentage of input data. Possible options are 025, 05, 1, 2, 4, 8, and full (24). Default to full.
        variant (bool): Determine the data to plot.
            True: variant dataset
            False: ML dataset
    """

    fig, ax = plt.subplots(1, 1, figsize=plot_utils.latex_figure(ratio=(16, 9)))

    data = dataset.epsilon_sigma_dataset(train_data=train_data)

    colour_map = {
        r"$\sigma \leq 1$ MeV": "#FBD900",
        r"$1 < \sigma \leq 2$ MeV": "#00AAF0",
        r"$2 < \sigma \leq 3$ MeV": "#098000",
        r"$\sigma \geq 4$ MeV": "#FF1503",
    }

    hue_order = [
        r"$\sigma \leq 1$ MeV",
        r"$1 < \sigma \leq 2$ MeV",
        r"$2 < \sigma \leq 3$ MeV",
        r"$\sigma \geq 4$ MeV",
    ]

    if variant == True:
        hue_data = "sigma_t"
    else:
        hue_data = "sigma"

    sns.scatterplot(
        data=data,
        x="N",
        y="Z",
        hue=hue_data,
        palette=colour_map,
        s=1.8,
        hue_order=hue_order,
        edgecolor="none",
    )

    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    ax.legend(title="", loc="lower right", markerscale=3)

    plot_utils.savefig(fig, ax, path)


def plot_old_data(path_exp, path_ext) -> None:
    """Plot the RMS Dev vs Number of Variants figure of the ML model incorporating variants' parameters. EXP refers to ML model predicting experimental masses, while EXT referes to ML model predicting extrapolated masses.

    Args:
        path_exp (str): path to save the EXP figure
        path_ext (str): path to save the EXT figure
    """

    data = pd.read_csv("data/result/old_run.csv")

    for data_type, ratio, save_path in [
        ["EXP", [1, 3, 4], path_exp],
        ["EXT", [1, 2, 4], path_ext],
    ]:
        fig, ax = plt.subplots(figsize=plot_utils.latex_figure(ratio=(9, 5)))

        for i in ratio:
            filtered_data = data[
                (data["dataset_type"] == data_type)
                & (data["var_predict"] / data["var_train"] == i)
            ]

            ax.scatter(
                filtered_data["var_train"], filtered_data["rms_dev"], label=f"1:{i}"
            )

        ax.set_xlabel("Number of Variants")
        ax.set_ylabel("RMS Deviation (MeV)")
        ax.set_ylim(bottom=0)

        ax.legend()

        plot_utils.savefig(fig, ax, save_path)


def plot_eps_dist_weight(path) -> None:
    """Plot the epsilon distribution of light and heavy nuclei

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(1, 2, figsize=plot_utils.latex_figure(ratio=(18, 8)))

    grouped = pd.read_parquet("data/result/full_mt_grouped.parquet")
    grouped["diff"] = (grouped["prediction_mean"] - grouped["target_mean"]) / grouped[
        "target_std"
    ]
    grouped["A"] = grouped["Z"] + grouped["N"]

    heavy_nuclei = grouped[grouped["A"] > 50]["diff"]
    light_nuclei = grouped[grouped["A"] <= 50]["diff"]
    # Plot each histogram on a different subplot
    sns.histplot(heavy_nuclei, stat="percent", ax=ax[0])
    ax[0].set_title("Heavy Nuclei")
    ax[0].set_xlabel("")

    sns.histplot(light_nuclei, stat="percent", ax=ax[1])
    ax[1].set_title("Light Nuclei")
    ax[1].set_ylabel("")
    ax[1].set_xlabel("")

    fig.supxlabel(r"$\epsilon$")

    print(f"Percentage Heavy: {heavy_nuclei.shape[0] / grouped.shape[0] * 100:.2f}")
    print(f"Percentage Light: {light_nuclei.shape[0] / grouped.shape[0] * 100:.2f}")
    print(
        f"IQR_Heavy Nuclei: {heavy_nuclei.quantile(0.75) - heavy_nuclei.quantile(0.25):.3f}"
    )
    print(
        f"IQR_Light Nuclei: {light_nuclei.quantile(0.75) - light_nuclei.quantile(0.25):.3f}"
    )

    plot_utils.savefig(fig, ax, path)


def plot_eps_dist_magic(path: str) -> None:
    """Plot the epsilon distribution of magic and tradtional magic nuclei

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(
        1, 2, figsize=plot_utils.latex_figure(ratio=(18, 8)), sharey=True
    )

    grouped = pd.read_parquet("data/result/full_mt_grouped.parquet")
    grouped["diff"] = (grouped["prediction_mean"] - grouped["target_mean"]) / grouped[
        "target_std"
    ]
    grouped["A"] = grouped["Z"] + grouped["N"]

    magic_N = [2, 8, 14, 20, 28, 40, 50, 82, 126, 184]
    magic_Z = [2, 8, 20, 28, 50, 82, 126]

    traditional_magic_Z = [2, 8, 20, 28, 50, 82]
    traditional_magic_N = [2, 8, 20, 28, 50, 82, 126]

    magic_nuclei = grouped[(grouped["N"].isin(magic_N)) | (grouped["Z"].isin(magic_Z))][
        "diff"
    ]
    traditional_magic_nuclei = grouped[
        (grouped["N"].isin(traditional_magic_N))
        | (grouped["Z"].isin(traditional_magic_Z))
    ]["diff"]

    print(f"Percentage Magic: {magic_nuclei.shape[0] / grouped.shape[0] * 100:.2f}")
    print(
        f"Percentage Traditional Magic: {traditional_magic_nuclei.shape[0] / grouped.shape[0] * 100:.2f}"
    )

    # Plot each histogram on a different subplot
    sns.histplot(magic_nuclei, stat="percent", ax=ax[0])
    ax[0].set_title("Magic Nuclei")
    ax[0].set_xlabel("")

    sns.histplot(traditional_magic_nuclei, stat="percent", ax=ax[1])
    ax[1].set_title("Traditional Magic Nuclei")
    ax[1].set_ylabel("")
    ax[1].set_xlabel("")

    fig.supxlabel(r"$\epsilon$")

    print(
        f"IQR_Magic Nuclei: {magic_nuclei.quantile(0.75) - magic_nuclei.quantile(0.25):.3f}"
    )
    print(
        f"IQR_Traditional Magic Nuclei: {traditional_magic_nuclei.quantile(0.75) - traditional_magic_nuclei.quantile(0.25):.3f}"
    )

    plot_utils.savefig(fig, ax, path)


def plot_eps_dist_magic_distance(path: str) -> None:
    """Plot the epsilon distribution of nucleons' distance from magic number, where epsilon > 1

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(
        1, 2, figsize=plot_utils.latex_figure(ratio=(18, 8)), sharey=True
    )

    # Load data
    data = pd.read_parquet("data/result/full_mt_grouped.parquet")
    # data = data []
    magic_number = [8, 20, 28, 50, 82, 126]
    data["eps"] = (data["target_mean"] - data["prediction_mean"]) / data["target_std"]
    data["dist_N"] = abs(data["N"].values[:, None] - magic_number).min(axis=1)
    data["dist_Z"] = abs(data["Z"].values[:, None] - magic_number).min(axis=1)
    filtered_data = data[abs(data["eps"]) > 1]

    # Plot
    sns.histplot(filtered_data["dist_N"], stat="percent", ax=ax[0], binwidth=4)
    ax[0].set_xlabel(r"$\Delta N_m$")

    sns.histplot(filtered_data["dist_Z"], stat="percent", ax=ax[1], binwidth=1)
    ax[1].set_xlabel(r"$\Delta Z_m$")
    ax[1].set_ylabel("")

    plot_utils.savefig(fig, ax, path)


def plot_param_correlation(path: str) -> None:
    """Plot the correlation of parameters with some quantities.

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure())

    moment_df = pd.read_parquet("data/result/sorted_variants_rms.parquet")
    variant_id = pd.read_parquet("data/input/bsk24_variants_ext.parquet")

    skyrme_param = pd.concat(
        [variant_id["varian_id"], variant_id.iloc[:, 2:23]], axis=1
    )
    skyrme_param.rename(columns={"varian_id": "variant_id"}, inplace=True)

    # Set x_2 to t_2.x_2
    skyrme_param["param(09)"] = skyrme_param["param(09)"] * skyrme_param["param(03)"]

    # Set param to relative param i.e. (param_variant - param_bsk)
    skyrme_param.iloc[:, 1:] = skyrme_param.apply(
        dataset.relative_skyrme, axis=1, result_type="expand"
    )

    moment_n_skyrme = pd.merge(
        moment_df[["variant_id", "rms_dev", "r_std", "variant_rms", "ml_rms"]],
        skyrme_param,
        on="variant_id",
        how="inner",
    )

    # Correlation test
    target_columns = ["rms_dev", "r_std", "variant_rms", "ml_rms"]
    result_columns = [col for col in moment_n_skyrme.columns if col.startswith("param")]

    corr_matrix = pd.DataFrame(
        index=target_columns, columns=result_columns, dtype=float
    )

    for index in target_columns:
        for column in result_columns:
            corr = moment_n_skyrme[[index, column]].corr(method="spearman").iloc[0, 1]
            corr_matrix.loc[index, column] = round(corr, 3)

    # display(corr_matrix)

    column_name = [
        r"$t_0$",
        r"$t_1$",
        r"$t_2$",
        r"$t_3$",
        r"$t_4$",
        r"$t_5$",
        r"$x_0$",
        r"$x_1$",
        r"$t_2 x_2$",
        r"$x_3$",
        r"$x_4$",
        r"$x_5$",
        r"$\alpha$",
        r"$\beta$",
        r"$\gamma$",
        r"$W_0$",
        r"$f_n^+$",
        r"$f_n^-$",
        r"$f_p^+$",
        r"$f_p^-$",
        r"$\epsilon_\Lambda$",
    ]

    index_name = [
        r"$\mu^\text{RMSE}$",
        r"$r_\sigma$",
        r"$\mu_v^\text{RMS}$",
        r"$\mu_\text{ML}^\text{RMS}$",
    ]

    corr_matrix.columns = column_name
    corr_matrix.index = index_name

    # Reshape the DataFrame from wide → long (for plotting)
    corr_matrix_long = corr_matrix.reset_index().melt(
        id_vars="index", var_name="data", value_name="value"
    )
    corr_matrix_long = corr_matrix_long.rename(columns={"index": "parameter"})

    sns.barplot(data=corr_matrix_long, x="data", y="value", hue="parameter", ax=ax)

    ax.set_ylabel("Correlation")
    ax.set_xlabel("")
    plt.xticks(rotation=45)
    ax.legend(loc="lower left")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pairing_str_diff(path: str) -> None:
    """Plot the distribution of proton and neutron pairing strength

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure(fraction=0.7))

    data = pd.read_parquet("data/input/bsk24_variants_ext.parquet")
    df_n = data["param(18)"] - data["param(17)"]
    df_p = data["param(20)"] - data["param(19)"]

    sns.histplot(df_n, stat="percent", label="$\Delta f_n$", ax=ax)
    sns.histplot(df_p, stat="percent", label="$\Delta f_p$", ax=ax)

    ax.set_xlabel("$\Delta f$")
    plt.legend()

    plot_utils.savefig(fig, ax, path)


def plot_goriely_uncertainty(path: str) -> None:
    """Plot the uncertainty of the mass prediction from Goriely et al's study

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure())

    backward_mc_data = dataset.goriely_uncertainty("backward")
    forward_mc_data = dataset.goriely_uncertainty("forward")

    colour_map = {
        r"$\sigma \leq 1$ MeV": "#FBD900",
        r"$1 < \sigma \leq 2$ MeV": "#00AAF0",
        r"$2 < \sigma \leq 3$ MeV": "#098000",
        r"$\sigma \geq 4$ MeV": "#FF1503",
    }

    hue_order = [
        r"$\sigma \leq 1$ MeV",
        r"$1 < \sigma \leq 2$ MeV",
        r"$2 < \sigma \leq 3$ MeV",
        r"$\sigma \geq 4$ MeV",
    ]

    sns.scatterplot(
        data=forward_mc_data,
        x="N",
        y="Z",
        hue="sigma",
        palette=colour_map,
        s=1.8,
        hue_order=hue_order,
        edgecolor="none",
    )

    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    ax.legend(title="", loc="lower right", markerscale=3)

    plot_utils.savefig(fig, ax, path)


def plot_computational_cost_ch3(path: str) -> None:
    """Plot the computational cost vs number of training datset

    Args:
        path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=plot_utils.latex_figure())

    data = pd.read_csv("data/summary/optimum_data_test.csv")
    data["training_data_percentage"] = data["run_name"].apply(
        lambda name: float(name.split("_")[-1]) * 0.8 * 100
    )

    data["run_time"] = data["run_time"].apply(convert_time_to_s)

    sns.boxplot(
        x="training_data_percentage",
        y="run_time",
        data=data,
        # fliersize=3,
        # width=0.5,
        ax=ax,
    )

    # Adjust ticks to show only half of them
    ticks = ax.get_xticks()
    labels: list[str] = [item.get_text()[:4] for item in ax.get_xticklabels()]

    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2])

    plot_utils.savefig(fig, ax, path)


def convert_time_to_s(time):
    hour = float(time.split(":")[0]) * 3600
    minute = float(time.split(":")[1]) * 60
    second = float(time.split(":")[2])

    return hour + minute + second


def plot_cost_full_data(path: str) -> None:
    """Plot the computational cost vs number of training datset

    Args:
        path (str): path to save the figure
    """

    fig, ax = plt.subplots(figsize=plot_utils.latex_figure())

    files = [p.name for p in Path("data/summary").glob("full_scale*") if p.is_file()]
    training_data_percentage = []
    avg_run_time = []
    avg_dev = []

    for file in files:
        if file == "full_scale.csv":
            training_data_percentage.append("24")
        else:
            training_data_percentage.append(file.split(".")[0].split("_")[-1])

        data = pd.read_csv("data/summary/" + file)
        run_time = data["run_time"].apply(convert_time_to_s)
        avg_run_time.append(np.mean(run_time))
        avg_dev.append(data["rms_dev"].mean())

    training_data_percentage = [
        ("0." + s[1:]) if s.startswith("0") else s for s in training_data_percentage
    ]
    training_data_percentage = [float(s) for s in training_data_percentage]

    ax.scatter(training_data_percentage, avg_dev)
    print(avg_dev)

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
