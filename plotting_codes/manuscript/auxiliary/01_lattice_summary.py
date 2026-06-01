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
WC_1nn_CrCr = np.load("data/WC_params/WC_avg.npy")[0, 1:-3, 0]  # (3, 17, 6) -> (13,)
WC_1nn_sum = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K

# Temperature color map
Viridis = plt.colormaps["viridis"].reversed()
cmap = ScalarMappable(Normalize(min(WC_1nn_sum), max(WC_1nn_sum)), Viridis)

os.system("mkdir -p figures/")

################################################################################
# Plot lattice temperature evolution.                                          #
################################################################################

# Start figure.
fig, ax = plt.subplots(figsize=(2.8 * 1.23, 2.6))

# Plot.
for idx, WC in enumerate(WC_1nn_sum):
    ax.plot(TE_list, xlattice[idx, :], "-o", c=cmap.to_rgba(WC))

# Add colorbar with specific ticks
tick_values = np.linspace(min(WC_1nn_sum), max(WC_1nn_sum), 6)  # 6 ticks between min and max
cbar = fig.colorbar(
    cmap,
    ax=ax,
    pad=0.03,
    ticks=tick_values,
    aspect=60,
    fraction=1,
    label=r"$\alpha^{total}$",
    orientation="vertical",
)
cbar.ax.tick_params(labelsize=5)
# Format tick labels to show 3 decimal places
cbar.ax.set_yticklabels([f"{x:.2f}" for x in tick_values])

# Add details.
ax.set_ylabel(r"Lattice parameter ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)

# Save figure.
fig.savefig("figures/lattice_summary.pdf")
plt.close()
