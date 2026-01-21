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

# analysis.plot_report.plot_me_bsk_comparison(
#     "../5_Writing/chapters/1_introduction/image/deviation.pdf"
# )
analysis.plot_report.plot_loss_convergence(
    "../5_Writing/chapters/3_hyperparam_tuning/image/loss_convergence.pdf"
)
# analysis.plot_report.plot_gap_n_asymmetry()
# analysis.plot_report.plot_sigma_propagation()
# analysis.plot_report.plot_pure_uncertainty()
