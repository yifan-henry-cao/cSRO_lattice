import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator

from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.optimize import root_scalar

################################################################################
# Load and process data.                                                       #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
T_fit = np.linspace(400, 1600, 101)

# Load WC data
WC_1nn_sum = np.sum(np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1) # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K, Cr-Cr
WC_spline = CubicSpline(T_list, WC_1nn_sum)

################################################################################
# Plot lattice temperature evolution.                                          #
################################################################################

# Start figure.
fig, ax = plt.subplots(figsize=(3.5 * 0.8, 2.69))

# Plot experimental data.
ax.plot(T_list, WC_1nn_sum, "o", c="#216b7b")
ax.plot(T_fit, WC_spline(T_fit), "-", c="#216b7b", label="Equilibrium CSRO")

# Add details.
ax.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
# ax.set_xlim(0, 2000)
ax.legend(loc='best')

# Save figure.
fig.savefig("figures/alpha_vs_T.png", dpi=300, transparent=False)

################################################################################
# Check spline fit for the alpha parameter.                                    #
################################################################################


# plt.close()

# # Start figure.
# fig, ax = plt.subplots(figsize=(3.5, 2.69))

# # Plot experimental data.
# ax.plot(T_list, WC_1nn_CrCr, "C0o", label="Raw data")
# T_fit = np.linspace(400, 1600, 101)
# ax.plot(T_fit, WC_spline(T_fit), "C1-", label="Cubic Spline")
# # ax.plot(T2, TSRO2, "C1-", label="2nd Heating")
# # ax.plot(TE_list, TE_list, "k--", linewidth=1)

# # Add details.
# ax.set_ylabel(r"WC", fontsize=8)
# ax.set_xlabel("Temperature (K)", fontsize=8)
# # ax.set_xlim(0, 2000)
# ax.legend()

# # Save figure.
# fig.savefig("figures/alpha_fit.png", dpi=300, transparent=False)
# plt.close()

################################################################################
# Plot lattice temperature evolution.                                          #
################################################################################

# # Start figure.
# fig, ax = plt.subplots(figsize=(3.5 * 1.3, 2.69))

# # Plot.
# for idx, T0 in enumerate(T_list):
#     ax.plot(TE_list, xlattice_fitted[idx, :], "-o", c=cmap.to_rgba(T0), alpha=0.9)

# # Plot experimental data.
# ax.plot(T1, lat1, "C0-", label="1st Heating")
# ax.plot(T2, lat2, "C1-", label="2nd Heating")

# # Add colorbar
# cbar = fig.colorbar(
#     cmap,
#     ax=ax,
#     pad=0.03,
#     ticks=T_list,
#     aspect=40,
#     fraction=1,
#     label="T0 (SRO)",
#     orientation="vertical",
# )
# cbar.ax.tick_params(labelsize=6)
# cbar.ax.yaxis.set_minor_locator(NullLocator())

# # Add details.
# ax.set_ylabel(r"Lattice parameter ($\mathring{\mathrm{A}}$)", fontsize=8)
# ax.set_xlabel("Temperature (K)", fontsize=8)
# # ax.set_xlim(0, 2000)
# ax.legend()

# # Save figure.
# fig.savefig("figures/fitted_lattice.png", dpi=300, transparent=False)
# plt.close()
