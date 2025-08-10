import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator


os.system("mkdir -p figures/")


################################################################################
# Load and process data.                                                       #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)
ID_list = np.array(range(1, 10))  # Independent runs

xlattice_data = np.load("data/sim_data/lattice.npy")
xlattice = mean(xlattice_data, axis=-1)  # (13, 33, 10) -> (13, 33)

from scipy.interpolate import RectBivariateSpline

# Generate mesh grid
x, y = np.meshgrid(T_list, TE_list)

# Perform 2D spline fit
spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)

x_fit = np.arange(400, 1201, 5)
y_fit = x_fit
z_equilibrium = spline(x_fit, y_fit, grid=False)

# 1073K fig 1a highest temperature from dilatometry (Thigh before 3rd heating)
# 1473K annealing temperature
x_fixed = np.ones_like(y_fit) * 1473
z_1473K = spline(x_fixed, y_fit, grid=False)

x_varying = np.array([884 if y < 884 else y for y in y_fit])
z_varying = spline(x_varying, y_fit, grid=False)

# Start figure.
# fig, ax = plt.subplots(figsize=(3.5 * 0.8, 2.69))
fig, ax = plt.subplots(figsize=(2.8 * 1.05, 2.6))

# Add details.
ax.set_ylabel(r"Lattice parameter ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_xlim(375, 1225)

# Plot experimental data.
# ax.plot(y_fit, z_1473K, "-", color="#9ed925", label="Low CSRO (1473K)", linewidth=1.5)
# ax.plot(y_fit, z_varying, "-", color="#37b35f", label="Evolving CSRO", zorder=9, linewidth=1.5)
# ax.plot(y_fit, z_equilibrium, "-", color="#216b7b", label="Equilibrium CSRO", linewidth=1.5)
ax.plot(y_fit, z_1473K, "-", color="#F7931E", label="Low CSRO (1473 K)", linewidth=1.5, alpha=0.9)
ax.plot(y_fit, z_varying, "-", color="#37b35f", label="Evolving CSRO", zorder=9, linewidth=1.5, alpha=0.9)
ax.plot(y_fit, z_equilibrium, "-", color="#6E96E5", label="Equilibrium CSRO", linewidth=1.5, alpha=0.9)
ax.legend(loc="upper left", fontsize=7)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig("figures/lattice_constants.png", dpi=300, transparent=False)
plt.close()
