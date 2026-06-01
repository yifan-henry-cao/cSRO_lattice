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

# Temperature color map - Red to Blue
RedtoBlue = LinearSegmentedColormap.from_list("RedtoBlue", ["#C00000", "#0070C0"])
cmap = ScalarMappable(Normalize(min(WC_1nn_sum), max(WC_1nn_sum)), RedtoBlue)

os.system("mkdir -p figures/")

################################################################################
# Plot lattice temperature evolution for selected CSRO temperatures.          #
################################################################################

idx_400K = np.where(T_list == 400)[0][0]
idx_800K = np.where(T_list == 800)[0][0]
idx_1600K = np.where(T_list == 1600)[0][0]

selected_indices = [idx_400K, idx_800K, idx_1600K]
selected_temps = [400, 800, 1600]

tick_values = np.linspace(min(WC_1nn_sum), max(WC_1nn_sum), 6)

y_min = np.min(xlattice)
y_max = np.max(xlattice)
y_padding = (y_max - y_min) * 0.05
y_lim = [y_min - y_padding, y_max + y_padding]

for main_idx, temp in zip(selected_indices, selected_temps):
    fig, ax = plt.subplots(figsize=(2.8 * 1.23, 2.6))

    for idx in range(len(T_list)):
        if idx != main_idx:
            WC = WC_1nn_sum[idx]
            ax.plot(TE_list, xlattice[idx, :], "-o", c=cmap.to_rgba(WC), alpha=0.2)

    WC_main = WC_1nn_sum[main_idx]
    ax.plot(TE_list, xlattice[main_idx, :], "-o", c=cmap.to_rgba(WC_main))

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
    cbar.ax.set_yticklabels([f"{x:.2f}" for x in tick_values])

    ax.set_ylabel(r"Lattice parameter ($\mathring{\mathrm{A}}$)", fontsize=10)
    ax.set_xlabel("Temperature (K)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_ylim(y_lim)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

    fig.savefig(f"figures/s06_lattice_vs_T_CSRO_{temp}K.pdf")
    plt.close(fig)
