import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator

################################################################################
# Load and process data.                                                       #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)

# Load lattice and SRO data
xlattice_data = np.load("data/sim_data/lattice.npy")  # (13, 33, 10)
xlattice = mean(xlattice_data, axis=-1)  # (13, 33, 10) -> (13, 33)
WC_1nn_sum = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K

# Temperature color map for TE
BluetoRed = LinearSegmentedColormap.from_list("BluetoRed", ["#1f77b4", "#d62728"])
cmap = ScalarMappable(Normalize(min(TE_list), max(TE_list)), BluetoRed)

os.system("mkdir -p figures/")

################################################################################
# Plot lattice change vs WC for all TE                                        #
################################################################################
lattice_random = np.zeros_like(TE_list).astype(np.float64)
lattice_adjusted = np.zeros_like(xlattice)

# Start figure.
fig, ax = plt.subplots(figsize=(3.5 * 1.3, 2.69))

for idx, TE in enumerate(TE_list):
    lattice_at_TE = xlattice[:, idx]

    # Fit linear regression
    coeffs = np.polyfit(WC_1nn_sum, lattice_at_TE, deg=1)
    slope, y_intercept = coeffs

    lattice_random[idx] = y_intercept

    # Subtract y-intercept and plot
    adjusted_lattice = lattice_at_TE - y_intercept
    lattice_adjusted[:, idx] = adjusted_lattice
    ax.plot(WC_1nn_sum, adjusted_lattice * 100, "-", color=cmap.to_rgba(TE), alpha=0.7, linewidth=1)

lattice_adjusted_mean = np.mean(lattice_adjusted, axis=1)
# Fit linear regression
coeffs = np.polyfit(WC_1nn_sum, lattice_adjusted_mean, deg=1)
slope, y_intercept = coeffs

alpha_fit = np.linspace(min(WC_1nn_sum), max(WC_1nn_sum), 101)
lattice_fit = slope * alpha_fit + y_intercept

ax.plot(alpha_fit, lattice_fit * 100, "--", color="black", linewidth=1.5, zorder=9, alpha=1)

# Add colorbar
cbar = fig.colorbar(cmap, ax=ax, pad=0.03, aspect=40, fraction=0.046, label="Temperature (K)", orientation="vertical")
cbar.ax.tick_params(labelsize=6)

# Add text with equation
equation_text = (
    rf"$a(T, \alpha^{{total}})$ = $a^{{0}}(T)$ - $\alpha^{{total}}\times${-slope:.2E} $\mathring{{\mathrm{{A}}}}$"
)
plt.text(0.05, 0.10, equation_text, transform=plt.gca().transAxes, fontsize=8, verticalalignment="top")

# Add details
ax.set_xlabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_ylabel(r"Lattice change due to CSRO (pm)", fontsize=8)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig("figures/s02_lattice_vs_alpha.pdf")
plt.close()

################################################################################
# Plot lattice baseline vs temperature                                        #
################################################################################

# Start figure.
fig, ax = plt.subplots(figsize=(3.5 * 1.0, 2.69))

# Fit quadratic function for random lattice
coeffs, residuals, _, _, _ = np.polyfit(TE_list, lattice_random, deg=2, full=True)
A_coeff, B_coeff, C_coeff = coeffs

quadratic_fit = lambda T: A_coeff * T**2 + B_coeff * T + C_coeff

T_fit = np.linspace(min(TE_list), max(TE_list), 101)
ax.plot(T_fit, quadratic_fit(T_fit), "-", color="black", linewidth=1, zorder=1)
for idx, TE in enumerate(TE_list):
    ax.plot(TE, lattice_random[idx], "o", color=cmap.to_rgba(TE), markersize=3, zorder=2)

# Add text with equation
equation_text = rf"""$a^{{0}}(T)$ = {C_coeff:.3f} + $T\times${B_coeff:.2E}
            + $T^2\times${A_coeff:.2E} $\mathring{{\mathrm{{A}}}}$"""
plt.text(0.05, 0.97, equation_text, transform=plt.gca().transAxes, fontsize=8, verticalalignment="top")

# Add details
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Lattice parameter baseline $a^{0}$ ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig("figures/s02_lattice_baseline.pdf")
plt.close()
