import matplotlib
import matplotlib.pyplot as plt

import analysis

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "figure.constrained_layout.use": True,  # tight layout
        "lines.markersize": 3.5,
        "lines.linewidth": 1.5,
    }
)

# analysis.plot_report.plot_correlation_illustration(
#     "../master-thesis/chapters/2_theory/image/correlation_illustration.pdf"
# )

# analysis.plot_report.plot_me_bsk_comparison(
#     "../5_Writing/chapters/1_introduction/image/deviation.pdf"
# )
# analysis.plot_report.plot_loss_convergence(
#     "../5_Writing/chapters/3_hyperparam_tuning/image/loss_convergence.pdf"
# )

# analysis.plot_report.plot_robustness_samebase(
#     "../master-thesis/chapters/3_hyperparam_tuning/image/robustness_samebase.pdf"
# )
#
# analysis.plot_report.plot_robustness_diffbase(
#     "../master-thesis/chapters/3_hyperparam_tuning/image/robustness_diffbase.pdf"
# )
#
# analysis.plot_report.plot_stability_one(
#     "../master-thesis/chapters/3_hyperparam_tuning/image/stability_one.pdf"
# )
#
# analysis.plot_report.plot_percent_train(
#     "../master-thesis/chapters/3_hyperparam_tuning/image/percent_train.pdf"
# )
#
# analysis.plot_report.plot_ic50_analysis(
#     "../master-thesis/chapters/3_hyperparam_tuning/image/ic50_analysis.pdf"
# )
#
# analysis.plot_report.plot_rmse_dist(
#     "../master-thesis/chapters/4_full_scale/image/rmse_dist.pdf"
# )
#
# analysis.plot_report.plot_moment_correlation(
#     "../master-thesis/chapters/4_full_scale/image/moment_corr.pdf"
# )
#
# analysis.plot_report.plot_delta_mu(
#     "../master-thesis/chapters/4_full_scale/image/fullscale_dmu.pdf"
# )
#
# analysis.plot_report.plot_rstd(
#     "../master-thesis/chapters/4_full_scale/image/fullscale_rsigma.pdf"
# )

analysis.plot_report.plot_epsilon(
    "../master-thesis/chapters/4_full_scale/image/fullscale_epsilon.pdf"
)
