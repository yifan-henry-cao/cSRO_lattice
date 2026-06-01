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

y_fit = np.arange(400, 1201, 5)
x_fit = np.array([733 if y < 733 else y for y in y_fit])
z_equilibrium = spline(x_fit, y_fit, grid=False)

# 1073K fig 1a highest temperature from dilatometry (Thigh before 3rd heating)
# 1473K annealing temperature
x_fixed = np.array([870 if y < 870 else y for y in y_fit])
z_870K = spline(x_fixed, y_fit, grid=False)

x_varying = np.array([904 if y < 904 else y for y in y_fit])
z_904K = spline(x_varying, y_fit, grid=False)

# Start figure.
fig, ax = plt.subplots(figsize=(3.5, 2.69))

# Add details.
ax.set_ylabel(r"Lattice parameter ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_xlim(375, 1225)

# Plot.
ax.plot(y_fit, z_equilibrium, "-", color="#216b7b", label=r"T$_{kr}$=733K", linewidth=1.5, alpha=0.7)
ax.plot(y_fit, z_870K, "-", color="#37b35f", label=r"T$_{kr}$=870K", linewidth=1.5, alpha=0.7)
ax.plot(y_fit, z_904K, "-", color="#9ed925", label=r"T$_{kr}$=904K", zorder=9, linewidth=1.5, alpha=0.7)
ax.legend(loc="upper left", fontsize=7)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig("figures/s04_lattice_constants_Tkr.pdf")
plt.close()
