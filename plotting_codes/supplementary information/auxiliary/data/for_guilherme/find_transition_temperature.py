import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")


# --- Load experimental volumetric data and transform to lattice constant ---
def load_volumetric_lattice():
    data_dir = "./"
    files = [
        "Vol_vs_T_h1.csv",
        "Vol_vs_T_h2.csv",
        "Vol_vs_T_h3.csv",
    ]
    # labels = ["1st Heating", "2nd Heating", "3rd Heating"]
    labels = ["Aged state", "Re-heated state", "Quenched state"]
    # Colorblind-friendly: blue, green, vermilion
    colors = ["#216b7b", "#37b35f", "#9ed925"]
    a0 = 3.6093  # at 1070 K
    v0 = 133.113043115209e21  # A^3
    natom = v0 / (a0**3 / 4)
    fcc_atoms_per_cell = 4
    mm3_to_A3 = 1e21
    T_all = []
    a_all = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        data = np.genfromtxt(path, delimiter=",", skip_header=2, usecols=(0, 1))
        T = data[:, 0]
        V = data[:, 1]
        V_A3 = V * mm3_to_A3
        V_per_atom = V_A3 / natom
        V_fcc = V_per_atom * fcc_atoms_per_cell
        a = V_fcc ** (1 / 3)
        T_all.append(T)
        a_all.append(a)
    return T_all, a_all, labels, colors


def find_transition_temperature(T, alpha, min_idx=1200, max_idx=2200):
    # Only consider interior points for transition
    best_idx = min_idx
    best_score = np.inf
    for idx in range(min_idx, max_idx):
        # Fit left
        p_left = np.polyfit(T[:idx], alpha[:idx], 1)
        fit_left = np.polyval(p_left, T[:idx])
        # Fit right
        p_right = np.polyfit(T[idx:], alpha[idx:], 1)
        fit_right = np.polyval(p_right, T[idx:])
        # Sum of squared residuals
        score = np.sum((alpha[:idx] - fit_left) ** 2) + np.sum((alpha[idx:] - fit_right) ** 2)
        if score < best_score:
            best_score = score
            best_idx = idx
    return T[best_idx], best_idx


# --- Load experimental lattice parameter data ---
T_all, a_all, exp_labels, exp_colors = load_volumetric_lattice()

fig, ax = plt.subplots(figsize=(2.8, 2.6))

for i, (T, a, color, label) in enumerate(zip(T_all, a_all, exp_colors, exp_labels)):
    # Find transition temperature for each sample
    # Use a as the y-data for transition finding
    # Use a reasonable min_idx and max_idx for the data length
    min_idx = int(0.2 * len(T))
    max_idx = int(0.8 * len(T))
    T_trans, idx_trans = find_transition_temperature(T, a, min_idx=min_idx, max_idx=max_idx)
    print(f"Transition temperature for {label}: {T_trans}K")
    # Left of transition: semitransparent
    ax.plot(T[:idx_trans], a[:idx_trans], "-", c=color, alpha=0.5, linewidth=1.5)
    # Right of transition: solid, with label
    ax.plot(T[idx_trans:], a[idx_trans:], "-", c=color, alpha=0.7, linewidth=1.5, label=label)
    # Vertical line at transition, only up to a value
    ax.vlines(T_trans, ymin=0, ymax=a[idx_trans], color=color, linestyles="--", alpha=0.5, linewidth=0.8, zorder=0)
    # Mark the transition temperature with a red star
    ax.plot(T_trans, a[idx_trans], marker="*", color=color, markersize=3, zorder=10)
    # Annotate the transition temperature
    ax.annotate(
        f"{T_trans:.0f} K",
        (T_trans, a[idx_trans]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=7,
        color=color,
    )

ax.set_ylabel(r"Lattice parameter $a$ ($\mathrm{\AA}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_xlim([500, 1125])
# Set y-limits based on data range
all_a = np.concatenate(a_all)
ax.set_ylim([all_a.min() * 0.997, all_a.max() * 1.003])
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)
fig.savefig("./lattice_vs_T.png", dpi=300, transparent=False)
