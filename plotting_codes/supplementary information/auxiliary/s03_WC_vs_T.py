import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("paper")
from numpy import *

c = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]
cm = plt.get_cmap("gist_rainbow")

ncomponent = 3
element = ["Cr", "Co", "Ni"]
Temp_list = array(range(300, 1901, 100))  # Temperature

WC_avg = load("data/WC_params/WC_avg.npy")
WC_std = load("data/WC_params/WC_std.npy")
colors = ["#8c564b", "#90C47F", "#FE7C46", "#21B0FE", "#9069C5", "#FE218B"]
label_list = ["Cr-Cr", "Co-Cr", "Cr-Ni", "Co-Co", "Co-Ni", "Ni-Ni"]

fig, ax = plt.subplots(figsize=(3.5 * 1.0, 2.69))
pair_idx = 0
for icomp in range(ncomponent):
    for jcomp in range(icomp, ncomponent):
        # label_txt = element[icomp] + "-" + element[jcomp]
        label_txt = label_list[pair_idx]
        mean = WC_avg[0, :, pair_idx]
        std = WC_std[0, :, pair_idx]
        ax.plot(Temp_list, mean, "o-", color=colors[pair_idx], label=label_txt, markersize=2)
        ax.fill_between(Temp_list, mean - std, mean + std, facecolor=colors[pair_idx], alpha=0.3)
        pair_idx += 1
ax.axhline(0, color="black", linewidth=1, alpha=0.5, linestyle="--", zorder=0)

# Add details.
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel("CSRO parameter", fontsize=8)
ax.legend(loc="best", fontsize=6, ncol=2, frameon=True)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig(f"./figures/WC_vs_T.png", dpi=300, transparent=False)
plt.close()

# Prepare pair labels
pair_labels = []
for icomp in range(ncomponent):
    for jcomp in range(icomp, ncomponent):
        pair_labels.append(element[icomp] + "-" + element[jcomp])

# Prepare data: first column is Temp_list, others are WC_avg[0, :, pair_idx] for each pair
csv_data = np.column_stack([Temp_list] + [WC_avg[0, :, i] for i in range(len(pair_labels))])

header = ["Temperature"] + pair_labels
# os.makedirs("./figures", exist_ok=True)
np.savetxt("./data/for_guilherme/WC_vs_T.csv", csv_data, delimiter=",", header=",".join(header), comments="")
