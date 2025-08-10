import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("paper")

# Constants
a0 = 3.6093  # at 1070.? K
v0 = 133.113043115209e21  # A^3
natom = v0 / (a0**3 / 4)
print(natom)
# natom = 1.1324307633647949e22  # number of atoms
fcc_atoms_per_cell = 4
mm3_to_A3 = 1e21

# Data files
data_dir = "data"
files = [
    "Vol_vs_T_h1.csv",
    "Vol_vs_T_h2.csv",
    "Vol_vs_T_h3.csv",
]
labels = ["1st Heating", "2nd Heating", "3rd Heating"]
colors = ["C0", "C1", "C2"]

T_all = []
a_all = []

for fname in files:
    path = os.path.join(data_dir, fname)
    # Skip first two header rows
    data = np.genfromtxt(path, delimiter=",", skip_header=2, usecols=(0, 1))
    T = data[:, 0]
    V = data[:, 1]  # mm^3
    # Convert to lattice constant (A)
    V_A3 = V * mm3_to_A3
    V_per_atom = V_A3 / natom
    V_fcc = V_per_atom * fcc_atoms_per_cell
    a = V_fcc ** (1 / 3)
    T_all.append(T)
    a_all.append(a)

# Plot a vs T
fig, ax = plt.subplots(figsize=(3.5 * 1.0, 2.69))
for i in range(3):
    ax.plot(T_all[i], a_all[i], label=labels[i], color=colors[i], alpha=0.8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Lattice constant $a$ ($\mathring{\mathrm{A}}$)", fontsize=8)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/process_figures/a_vs_T_exp.png", dpi=300, transparent=False)


def compute_CTE_windowed(T_list, xlattice, window_size=21):
    """
    Compute local CTE using a moving window linear fit.
    For each point (except at the edges), fit a line to a window of points around it.
    """
    n = len(T_list)
    half_window = window_size // 2
    CTE = np.full(n, np.nan)
    for i in range(n):
        # Determine window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        if end - start < 3:
            continue  # Need at least 3 points to fit
        T_win = T_list[start:end]
        a_win = xlattice[start:end]
        # Linear fit
        p = np.polyfit(T_win, a_win, 1)
        da_dT = p[0]
        a_center = xlattice[i]
        CTE[i] = (da_dT / a_center) * 1e6  # ppm/K
    return T_list, CTE


# Compute and plot CTE
fig2, ax2 = plt.subplots(figsize=(3.5 * 1.0, 2.69))
window_size = 31  # You can adjust this as needed
for i in range(3):
    T_central, CTE = compute_CTE_windowed(T_all[i], a_all[i], window_size=window_size)
    ax2.plot(T_central, CTE, label=labels[i], color=colors[i], alpha=0.8)
ax2.set_xlabel("Temperature (K)", fontsize=8)
ax2.set_ylabel(r"Thermal Expansion Coefficient $\alpha$ (ppm/K)", fontsize=8)
ax2.legend(fontsize=7)
fig2.tight_layout()
fig2.savefig("figures/process_figures/CTE_vs_T_exp.png", dpi=300, transparent=False)
