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

# Chapter 1
# analysis.plot_report.plot_me_bsk_comparison(
#     "../master-thesis/chapters/1_introduction/image/deviation.pdf"
# )
#
# # Chapter 2
# analysis.plot_report.plot_correlation_illustration(
#     "../master-thesis/chapters/2_theory/image/correlation_illustration.pdf"
# )
#
# analysis.plot_report.plot_bsk_res(
#     "../master-thesis/chapters/2_theory/image/bsk_res.pdf"
# )

# Chapter 3

# analysis.plot_report.plot_loss_convergence(
#     "../5_Writing/chapters/3_hyperparam_tuning/image/loss_convergence.pdf"
# )

# analysis.plot_report.plot_robustness_samebase(
#     "../master-thesis/chapters/3_hyperparam_tuning/image/robustness_samebase.pdf"
# )

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

# Chapter 4
# analysis.plot_report.plot_uncertainty(
#     "../master-thesis/chapters/5_uq_cost/image/unc_full.pdf"
# )
# analysis.plot_report.plot_uncertainty(
#     "../master-thesis/chapters/5_uq_cost/image/unc_full_variant.pdf",
#     variant=True,
# )
#
# for train_data in ["025", "05", "1", "2", "4", "8"]:
#     analysis.plot_report.plot_uncertainty(
#         f"../master-thesis/chapters/5_uq_cost/image/unc_{train_data}.pdf",
#         train_data=train_data,
#     )

# analysis.plot_report.plot_uncertainty_low_variants(
#     "../master-thesis/chapters/5_uq_cost/image/unc_100_variants.pdf", 100
# )
# analysis.plot_report.plot_uncertainty_low_variants(
#     "../master-thesis/chapters/5_uq_cost/image/unc_500_variants.pdf", 500
# )

# Chapter 5
analysis.plot_report.plot_rmse_dist(
    "../master-thesis/chapters/4_full_scale/image/rmse_dist.pdf"
)

analysis.plot_report.plot_moment_correlation(
    "../master-thesis/chapters/4_full_scale/image/moment_corr.pdf"
)

analysis.plot_report.plot_variant_variability(
    "../master-thesis/chapters/4_full_scale/image/percent_outliers.pdf"
)

# analysis.plot_report.plot_delta_mu(
#     "../master-thesis/chapters/4_full_scale/image/fullscale_dmu.pdf"
# )
#
# analysis.plot_report.plot_rstd(
#     "../master-thesis/chapters/4_full_scale/image/fullscale_rsigma.pdf"
# )
#
# analysis.plot_report.plot_epsilon(
#     "../master-thesis/chapters/4_full_scale/image/fullscale_epsilon.pdf"
# )

# analysis.plot_report.plot_eps_dist_weight(
#     "../master-thesis/chapters/4_full_scale/image/eps_weight.pdf"
# )
#
# analysis.plot_report.plot_eps_dist_magic(
#     "../master-thesis/chapters/4_full_scale/image/eps_magic.pdf"
# )

# analysis.plot_report.plot_eps_dist_magic_distance(
#     "../master-thesis/chapters/4_full_scale/image/eps_magic_dist.pdf"
# )

# analysis.plot_report.plot_param_correlation(
#     "../master-thesis/chapters/4_full_scale/image/param_correlation.pdf"
# )

# analysis.plot_report.plot_pairing_str_diff(
#     "../master-thesis/chapters/4_full_scale/image/pairing_str_diff.pdf"
# )

# Appendix
# analysis.plot_report.plot_old_data(
#     "../master-thesis/chapters/appendix/image/RMSE_vs_variants_EXP.pdf",
#     "../master-thesis/chapters/appendix/image/RMSE_vs_variants_EXT.pdf",
# )

# Test
# analysis.plot_report.plot_computational_cost_ch3("test.pdf")
# analysis.plot_report.plot_cost_full_data("test.pdf")
