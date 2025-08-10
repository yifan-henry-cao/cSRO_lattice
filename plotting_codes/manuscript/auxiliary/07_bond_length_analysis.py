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
# bond_length_TE = np.load("data/sim_data/bond_length_TE.npy")  # (TE, ID, 3, 3)
bond_length_TE = np.load("data/sim_data/bond_length_TE_RSS.npy")  # (TE, ID, 3, 3), (33, 40, 3, 3)
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
b0_mean = np.mean(np.load("data/sim_data/bond_length_TE_RSS.npy"), axis=1)[24]  # (33, 40, 3, 3) -> (3, 3) at 1000K
b0_std = np.std(np.load("data/sim_data/bond_length_TE_RSS.npy"), axis=1)[24]  # (33, 40, 3, 3) -> (3, 3) at 1000K
WC_1nn_sum = np.append(WC_1nn_sum, [0])

# Plot each pair
idx = 0
for i in range(3):
    for j in range(i, 3):
        mean = np.append(bond_mean_TSRO[:, i, j], b0_mean[i, j])
        std = np.append(bond_std_TSRO[:, i, j], b0_std[i, j])
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

# # Save Figure 2 data to CSV files (mean and std)

# # Prepare data arrays for CSV
# mean_data = np.zeros((len(WC_1nn_sum), 7))  # 1 alpha + 6 pairs
# std_data = np.zeros((len(WC_1nn_sum), 7))
# mean_data[:, 0] = WC_1nn_sum
# std_data[:, 0] = WC_1nn_sum

# idx = 0
# for i in range(3):
#     for j in range(i, 3):
#         mean_data[:, idx + 1] = bond_mean_TSRO[:, i, j]
#         std_data[:, idx + 1] = bond_std_TSRO[:, i, j]
#         idx += 1

# header = ["alpha"] + pair_labels
# os.system("mkdir -p data/for_guilherme/")
# np.savetxt(
#     "data/for_guilherme/bond_length_vs_alpha_mean.csv", mean_data, delimiter=",", header=",".join(header), comments=""
# )
# np.savetxt(
#     "data/for_guilherme/bond_length_vs_alpha_std.csv", std_data, delimiter=",", header=",".join(header), comments=""
# )
