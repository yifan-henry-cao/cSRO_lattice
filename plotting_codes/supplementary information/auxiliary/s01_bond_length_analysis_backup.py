import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import NullLocator

# Create output directory
os.system("mkdir -p figures/")

################################################################################
# Load data                                                                    #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)

# Load bond length data
bond_length_TE = np.load("data/sim_data/bond_length_TE.npy")  # (TE, ID, 3, 3)
bond_length_TSRO = np.load("data/sim_data/bond_length_TSRO.npy")  # (T, ID, 3, 3)

# Load Warren-Cowley parameter for Cr-Cr pairs
# WC_1nn_CrCr = np.load("data/WC_params/WC_avg.npy")[0, 1:-3, 0]  # (13,) - matches T_list length
WC_1nn_sum = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K

# Calculate mean and std of bond lengths across independent runs
bond_mean_TE = np.mean(bond_length_TE, axis=1)  # (TE, 3, 3)
print(bond_length_TE.shape)
bond_std_TE = np.std(bond_length_TE, axis=1)  # (TE, 3, 3)
bond_mean_TSRO = np.mean(bond_length_TSRO, axis=1)  # (T, 3, 3)
bond_std_TSRO = np.std(bond_length_TSRO, axis=1)  # (T, 3, 3)

# Define pair labels (for 3 components: Ni-Ni, Ni-Co, Ni-Cr, Co-Co, Co-Cr, Cr-Cr)
pair_labels = ["Ni-Ni", "Co-Ni", "Co-Co", "Cr-Ni", "Co-Cr", "Cr-Cr"][::-1]
colors = ["#8c564b", "#90C47F", "#FE7C46", "#21B0FE", "#9069C5", "#FE218B"]

################################################################################
# Figure 1: Bond lengths vs Temperature (TE)                                   #
################################################################################
fig, ax = plt.subplots(figsize=(3.5 * 1.0, 2.69))

# Plot each pair
idx = 0
for i in range(3):
    for j in range(i, 3):
        mean = bond_mean_TE[:, i, j]
        std = bond_std_TE[:, i, j]
        ax.plot(TE_list, mean, "-", color=colors[idx], label=pair_labels[idx])
        ax.fill_between(TE_list, mean - std, mean + std, facecolor=colors[idx], alpha=0.3)
        idx += 1

ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel("Bond length ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.legend(fontsize=6, ncol=2, loc="best", frameon=True)
ax.set_xlim(375, 1225)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

fig.savefig("figures/bond_length_vs_TE.png", dpi=300, transparent=False)
plt.close()

################################################################################
# Figure 2: Bond lengths vs Alpha (TSRO)                                       #
################################################################################
fig, ax = plt.subplots(figsize=(3.5 * 1.0, 2.69))

# Plot each pair
idx = 0
for i in range(3):
    for j in range(i, 3):
        mean = bond_mean_TSRO[:, i, j]
        std = bond_std_TSRO[:, i, j]
        ax.plot(WC_1nn_sum, mean, "o-", color=colors[idx], label=pair_labels[idx], markersize=2)
        ax.fill_between(WC_1nn_sum, mean - std, mean + std, facecolor=colors[idx], alpha=0.3)
        idx += 1

ax.set_xlabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_ylabel("Bond length ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.12, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)
# ax.legend(fontsize=6, ncol=2)

fig.savefig("figures/bond_length_vs_alpha.png", dpi=300, transparent=False)
plt.close()

# Save Figure 2 data to CSV files (mean and std)

# Prepare data arrays for CSV
mean_data = np.zeros((len(WC_1nn_sum), 7))  # 1 alpha + 6 pairs
std_data = np.zeros((len(WC_1nn_sum), 7))
mean_data[:, 0] = WC_1nn_sum
std_data[:, 0] = WC_1nn_sum

idx = 0
for i in range(3):
    for j in range(i, 3):
        mean_data[:, idx + 1] = bond_mean_TSRO[:, i, j]
        std_data[:, idx + 1] = bond_std_TSRO[:, i, j]
        idx += 1

header = ["alpha"] + pair_labels
os.system("mkdir -p data/for_guilherme/")
np.savetxt(
    "data/for_guilherme/bond_length_vs_alpha_mean.csv", mean_data, delimiter=",", header=",".join(header), comments=""
)
np.savetxt(
    "data/for_guilherme/bond_length_vs_alpha_std.csv", std_data, delimiter=",", header=",".join(header), comments=""
)


################################################################################
# Figure 3: Bond lengths distributions                                         #
################################################################################
def plot_bond_distribution(pair_key, all_temp_bonds, pair_name, figsize, plot_cbar=False):
    fig, ax = plt.subplots(figsize=figsize)

    # Find global min and max bond lengths for this pair to set consistent bins
    all_bonds = []
    for T0 in T_list:
        all_bonds.extend(all_temp_bonds[T0][pair_key])

    min_bond = np.min(all_bonds)
    max_bond = np.max(all_bonds)

    # Create consistent bins for all temperatures
    bins = np.linspace(min_bond, max_bond, 40)

    # Plot histogram for each temperature
    for T0 in T_list:
        bonds = all_temp_bonds[T0][pair_key]
        if len(bonds) > 0:
            counts, bin_edges = np.histogram(bonds, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.plot(bin_centers, counts, alpha=0.8, color=cmap.to_rgba(T0), linewidth=1)

        # Compute mean bond length
        mean_bond = np.mean(bonds)

        # Find the bin center closest to the mean
        idx_closest = np.argmin(np.abs(bin_centers - mean_bond))
        mean_y = counts[idx_closest]

        # Plot the dot at the mean location
        plt.plot(
            mean_bond, mean_y, marker="o", markersize=6, markerfacecolor=cmap.to_rgba(T0), markeredgecolor="black"
        )

    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Count")
    plt.title(f"{pair_name} Bond Length Distribution")
    ax.set_ylim(bottom=0, top=5600)

    # Add colorbar
    if plot_cbar:
        cbar = fig.colorbar(
            cmap,
            ax=ax,
            pad=0.03,
            ticks=T_list,
            aspect=40,
            fraction=1,
            label="T0 (SRO)",
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.set_minor_locator(NullLocator())
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.08)

    # Save the plot
    plt.savefig(f"figures/bond_dist_{pair_name}.png", dpi=300, bbox_inches="tight", transparent=False)
    plt.close()


# Load the raw bond length data
all_temp_bonds = np.load("data/bond_statistics/all_bond_lengths.npy", allow_pickle=True).item()

# Get temperature list and sort it
T_list = np.array(sorted(all_temp_bonds.keys()))

# Setup color map
RedtoBlue = LinearSegmentedColormap.from_list("RedtoBlue", ["#1f77b4", "#d62728"])
cmap = ScalarMappable(Normalize(min(T_list), max(T_list)), RedtoBlue)

# Define atom types
atom_types = ["Cr", "Co", "Ni"]
ncomponent = len(atom_types)

pair_key = (0, 0)
pair_name = "Cr-Cr"
plot_bond_distribution(pair_key, all_temp_bonds, pair_name, (3.5 * 0.9, 2.69), plot_cbar=False)

pair_key = (0, 1)
pair_name = "Cr-Co"
plot_bond_distribution(pair_key, all_temp_bonds, pair_name, (3.5 * 1.1, 2.69), plot_cbar=True)

pair_key = (0, 2)
pair_name = "Cr-Ni"
plot_bond_distribution(pair_key, all_temp_bonds, pair_name, (3.5 * 0.9, 2.69), plot_cbar=False)

pair_key = (1, 1)
pair_name = "Co-Co"
plot_bond_distribution(pair_key, all_temp_bonds, pair_name, (3.5 * 1.1, 2.69), plot_cbar=True)

pair_key = (1, 2)
pair_name = "Co-Ni"
plot_bond_distribution(pair_key, all_temp_bonds, pair_name, (3.5 * 0.9, 2.69), plot_cbar=False)

pair_key = (2, 2)
pair_name = "Ni-Ni"
plot_bond_distribution(pair_key, all_temp_bonds, pair_name, (3.5 * 1.1, 2.69), plot_cbar=True)
