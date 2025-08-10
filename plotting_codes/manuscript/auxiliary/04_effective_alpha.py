import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os
from scipy.optimize import minimize_scalar

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy.interpolate import RectBivariateSpline, CubicSpline

################################################################################
# Load and process data.                                                       #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)

# Load lattice and SRO data
xlattice_data = np.load("data/sim_data/lattice.npy")  # (13, 33, 10)
xlattice = mean(xlattice_data, axis=-1)  # (13, 33, 10) -> (13, 33)
# WC_1nn_CrCr = np.load("data/WC_params/WC_avg.npy")[0, 1:-3, 0] # (3, 17, 6) -> (13,)
WC_1nn_sum = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K
WC_spline = CubicSpline(T_list, WC_1nn_sum)
T_fit = np.linspace(400, 1200, 101)

# Temperature color map for TE
BluetoRed = LinearSegmentedColormap.from_list("BluetoRed", ["#1f77b4", "#d62728"])
cmap = ScalarMappable(Normalize(min(TE_list), max(TE_list)), BluetoRed)

os.system("mkdir -p figures/")


################################################################################
# Now apply the thermal expansion correction from 04                            #
################################################################################
# Compute thermal expansion correction
def compute_CTE_nonuniform(T_list, xlattice):
    # Simple finite difference for CTE
    dT = T_list[1:] - T_list[:-1]
    dL = xlattice[1:] - xlattice[:-1]
    T_avg = (T_list[1:] + T_list[:-1]) / 2
    L_avg = (xlattice[1:] + xlattice[:-1]) / 2
    CTE = (dL / dT) / L_avg * 1e6
    return T_avg, CTE


def compute_CTE(T_list, xlattice):
    dT = T_list[1:] - T_list[:-1]
    dL = xlattice[:, 1:] - xlattice[:, :-1]
    T_avg = (T_list[1:] + T_list[:-1]) / 2
    L_avg = (xlattice[:, 1:] + xlattice[:, :-1]) / 2
    CTE = dL / dT / L_avg * 1e6
    return T_avg, CTE


# --- Load experimental volumetric data and transform to lattice constant ---
def load_volumetric_lattice():
    data_dir = "data"
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


def compute_CTE_windowed(T_list, xlattice, window_size=31):
    n = len(T_list)
    half_window = window_size // 2
    CTE = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        if end - start < 3:
            continue
        T_win = T_list[start:end]
        a_win = xlattice[start:end]
        p = np.polyfit(T_win, a_win, 1)
        da_dT = p[0]
        a_center = xlattice[i]
        CTE[i] = (da_dT / a_center) * 1e6
    return T_list, CTE


def transform_xlattice(T_list, TE_list, xlattice):
    # Load experimental data
    # T1, lat1 = np.loadtxt("data/Francisco_h1.csv", unpack=True)
    # T2, lat2 = np.loadtxt("data/Francisco_h2.csv", unpack=True)
    T_exp_all, a_exp_all, _, _ = load_volumetric_lattice()

    # For compatibility with old code, assign T1, lat1, T2, lat2, T3, lat3
    T1, lat1 = T_exp_all[0], a_exp_all[0]
    T2, lat2 = T_exp_all[1], a_exp_all[1]
    T3, lat3 = T_exp_all[2], a_exp_all[2]

    # Get reference points
    # a0_exp = lat3[-1]  # Lattice at 1073.19K
    # a0_exp = (lat1[-1] + lat2[-1]) / 2
    a0_exp = lat3[-1]
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)
    a0_sim = initial_spline(1073.19, 1073.19)
    a_diff = a0_exp - a0_sim

    T1_avg, CTE1 = compute_CTE_windowed(T1[:1200], lat1[:1200])
    T2_avg, CTE2 = compute_CTE_windowed(T2[:1200], lat2[:1200])
    # T3_avg, CTE3 = compute_CTE_windowed(T3[:1200], lat3[:1200])
    T_avg, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)

    # breakpoint()
    # thermal_exp = (np.mean(CTE1[15:-15]) + np.mean(CTE2[15:-15]) + np.mean(CTE3[15:-15])) / 3
    # thermal_exp = (np.mean(CTE1[160:1061]) + np.mean(CTE2[160:1061])) / 2  # 600K -> 750K
    thermal_exp = np.mean(CTE2[160:1061])
    thermal_sim = np.mean(sim_CTE, axis=0)[8:15].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6
    print(thermal_diff)

    # Apply thermal correction to adjusted lattice
    thermal_lat_diff = thermal_diff * (TE_list - 1073)  # Intercept at 1073K
    xlattice_final = (xlattice + a_diff) * (1 + thermal_lat_diff)

    # print(f"a_diff: {a_diff[0][0]}")
    # print(f"thermal_diff: {thermal_diff}")
    return xlattice_final


def transform_xlattice_new_sample(T_list, TE_list, xlattice):
    # Load experimental data
    T_exp_all, a_exp_all, _, _ = load_volumetric_lattice()

    # For compatibility with old code, assign T1, lat1, T2, lat2, T3, lat3
    T1, lat1 = T_exp_all[0], a_exp_all[0]
    T2, lat2 = T_exp_all[1], a_exp_all[1]
    T3, lat3 = T_exp_all[2], a_exp_all[2]

    # Get reference points
    a0_exp = lat3[-1]  # Lattice at 1073.19K
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=3, ky=2, s=0)
    a0_sim = initial_spline(1073.19, 1073.19)
    a_diff = a0_exp - a0_sim

    # T1_avg, CTE1 = compute_CTE_windowed(T1[:1200], lat1[:1200])
    T2_avg, CTE2 = compute_CTE_windowed(T2[:1200], lat2[:1200])
    T3_avg, CTE3 = compute_CTE_windowed(T3[:1200], lat3[:1200])
    T_avg, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)

    # breakpoint()
    # thermal_exp = (np.mean(CTE1[15:-15]) + np.mean(CTE2[15:-15]) + np.mean(CTE3[15:-15])) / 3
    thermal_exp = np.mean(CTE2[160:1061])
    thermal_sim = np.mean(sim_CTE, axis=0)[8:15].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6
    print(thermal_diff)

    # Apply thermal correction to adjusted lattice
    thermal_lat_diff = thermal_diff * (TE_list - 1073)  # Intercept at 1075K
    xlattice_final = (xlattice + a_diff) * (1 + thermal_lat_diff)

    # print(f"a_diff: {a_diff[0][0]}")
    # print(f"thermal_diff: {thermal_diff}")
    return xlattice_final


def obtain_fitting_parameters(xlattice_transformed):
    # Get baseline (random) lattice parameters
    lattice_random = np.zeros_like(TE_list).astype(np.float64)
    lattice_adjusted = np.zeros_like(xlattice)

    for idx, TE in enumerate(TE_list):
        lattice_at_TE = xlattice_transformed[:, idx]

        # Fit linear regression
        coeffs = np.polyfit(WC_1nn_sum, lattice_at_TE, deg=1)
        slope, y_intercept = coeffs

        lattice_random[idx] = y_intercept

        # Subtract y-intercept and get adjusted lattice
        adjusted_lattice = lattice_at_TE - y_intercept
        lattice_adjusted[:, idx] = adjusted_lattice

    # Adjusted lattice mean to obtain constant c
    lattice_adjusted_mean = np.mean(lattice_adjusted, axis=1)
    # Fit linear regression
    coeffs = np.polyfit(WC_1nn_sum, lattice_adjusted_mean, deg=1)
    shrink_parameter_C, small_misfit_D = coeffs

    # # Perform linear fit to express baseline
    # coeffs = np.polyfit(TE_list, lattice_random, deg=1)
    # random_expansion_B, lattice_0K_A = coeffs
    # fitting_coeffs = [lattice_0K_A, random_expansion_B, shrink_parameter_C, small_misfit_D]

    # Fit quadratic function for random lattice
    coeffs, residuals, _, _, _ = np.polyfit(TE_list, lattice_random, deg=2, full=True)
    A_coeff, B_coeff, C_coeff = coeffs
    fitting_coeffs = [A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D]

    # print(fitting_coeffs)

    return fitting_coeffs


# --- Use new experimental data for all further analysis ---
T_exp_all, a_exp_all, exp_labels, exp_colors = load_volumetric_lattice()

# For compatibility with old code, assign T1, lat1, T2, lat2, T3, lat3
T1, lat1 = T_exp_all[0], a_exp_all[0]
T2, lat2 = T_exp_all[1], a_exp_all[1]
T3, lat3 = T_exp_all[2], a_exp_all[2]

# # --- Recompute alpha for all three curves ---
# xlattice_sample_12 = transform_xlattice(T_list, TE_list, xlattice)
# lattice_0K_A, random_expansion_B, shrink_parameter_C, small_misfit_D = obtain_fitting_parameters(xlattice_sample_12)
# alpha1 = (lattice_0K_A + random_expansion_B * T1 + small_misfit_D - lat1) / (-shrink_parameter_C)
# alpha2 = (lattice_0K_A + random_expansion_B * T2 + small_misfit_D - lat2) / (-shrink_parameter_C)

# # Perform separate transformation for new sample
# xlattice_new = transform_xlattice_new_sample(T_list, TE_list, xlattice)
# lattice_0K_A, random_expansion_B, shrink_parameter_C, small_misfit_D = obtain_fitting_parameters(xlattice_new)
# alpha3 = (lattice_0K_A + random_expansion_B * T3 + small_misfit_D - lat3) / (-shrink_parameter_C)

# --- Recompute alpha for all three curves ---
xlattice_sample_12 = transform_xlattice(T_list, TE_list, xlattice)
A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D = obtain_fitting_parameters(xlattice_sample_12)
alpha1 = (C_coeff + B_coeff * T1 + A_coeff * T1**2 + small_misfit_D - lat1) / (-shrink_parameter_C)
alpha2 = (C_coeff + B_coeff * T2 + A_coeff * T2**2 + small_misfit_D - lat2) / (-shrink_parameter_C)

# Perform separate transformation for new sample
xlattice_new = transform_xlattice_new_sample(T_list, TE_list, xlattice)
A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D = obtain_fitting_parameters(xlattice_new)
alpha3 = (C_coeff + B_coeff * T3 + A_coeff * T3**2 + small_misfit_D - lat3) / (-shrink_parameter_C)


# --- Helper: Find transition temperature for piecewise linear fit ---
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


# --- Plot all three curves with transition temperature and alpha shading ---
fig, ax = plt.subplots(figsize=(2.8, 2.6))
ax.plot(T_fit, WC_spline(T_fit), "-", c="#6B93E4", label="Equilibrium", zorder=9)

for i, (T, alpha, color, label) in enumerate(zip([T1, T2, T3], [alpha1, alpha2, alpha3], exp_colors, exp_labels)):
    T_trans, idx_trans = find_transition_temperature(T, alpha)
    if i == 0:
        idx_trans -= 240
    elif i == 2:
        idx_trans += 150
    print(f"Transition temperature for heating {i}: {T_trans}K")
    # Left of transition: semitransparent
    ax.plot(T[:idx_trans], alpha[:idx_trans], "-", c=color, alpha=0.3, linewidth=1.5)
    # Right of transition: solid, with label
    ax.plot(T[idx_trans - 1 :], alpha[idx_trans - 1 :], "-", c=color, alpha=1, linewidth=1.5, label=label)
    # Vertical line at transition, only up to alpha value
    # ax.vlines(T_trans, ymin=0, ymax=alpha[idx_trans], color=color, linestyles="--", alpha=0.5, linewidth=0.8, zorder=0)

# ax.plot(T_list[:-4], WC_1nn_sum[:-4], "o", c="#6B93E4", zorder=9)
ax.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_xlim([475, 1225])
ax.set_ylim([0.29, 0.95])
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)
fig.savefig("figures/effective_alpha.png", dpi=300, transparent=False)

# --- Compute effective temperatures for all three curves ---
T_search = np.linspace(400, 1600, 1200)
WC_values = WC_spline(T_search)
T_eff1 = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha1])
T_eff2 = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha2])
T_eff3 = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha3])

fig_eff, ax_eff = plt.subplots(figsize=(2.8, 2.6))
ax_eff.plot(T1, T_eff1, "-", c=exp_colors[0], linewidth=1.5, label=exp_labels[0])
ax_eff.plot(T2, T_eff2, "-", c=exp_colors[1], linewidth=1.5, label=exp_labels[1])
ax_eff.plot(T3, T_eff3, "-", c=exp_colors[2], linewidth=1.5, label=exp_labels[2])
ax_eff.plot([400, 1200], [400, 1200], "--", c="gray", alpha=0.5, label="Equilibrium")
ax_eff.set_xlabel("Temperature (K)", fontsize=8)
ax_eff.set_ylabel("Effective Temperature (K)", fontsize=8)
ax_eff.legend(loc="best")
ax_eff.spines["right"].set_visible(True)
ax_eff.spines["top"].set_visible(True)
ax_eff.yaxis.set_label_coords(-0.15, 0.5)
ax_eff.xaxis.set_label_coords(0.5, -0.08)
fig_eff.savefig("figures/effective_temperature.png", dpi=300, transparent=False)

# --- Process old Francisco datasets (h1 and h2) ---
print("\n=== Processing old Francisco datasets ===")


def transform_xlattice_old(T_list, TE_list, xlattice):
    # Load old experimental data
    T1_old, lat1_old = np.loadtxt("data/Francisco_h1.csv", unpack=True)
    T2_old, lat2_old = np.loadtxt("data/Francisco_h2.csv", unpack=True)

    # Get reference points using old data
    a0_exp = lat2_old[-1] - 0.0018  # Use h2 as reference, minus exp offset
    # a0_exp = lat1[-3]  # Use h1 as reference
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)
    a0_sim = initial_spline(1073.19, 1073.19)
    a_diff = a0_exp - a0_sim

    # Compute CTE for old data
    T1_avg, CTE1 = compute_CTE_nonuniform(T1_old[:50], lat1_old[:50])
    T2_avg, CTE2 = compute_CTE_nonuniform(T2_old[:50], lat2_old[:50])
    # T1_avg, CTE1 = compute_CTE_nonuniform(T1_old[:59], lat1_old[:59])
    # T2_avg, CTE2 = compute_CTE_nonuniform(T2_old[:59], lat2_old[:59])
    # :59 and 7:14 or :66 and 7:15
    T_avg, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)

    thermal_exp = (np.mean(CTE1[1:]) + np.mean(CTE2)) / 2
    thermal_sim = np.mean(sim_CTE, axis=0)[7:13].mean()
    # thermal_sim = np.mean(sim_CTE, axis=0)[7:14].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6
    print(f"Old data thermal_diff: {thermal_diff}")
    # breakpoint()

    # Apply thermal correction to adjusted lattice
    thermal_lat_diff = thermal_diff * (TE_list - 1073.19)  # Intercept at 1073K
    xlattice_final = (xlattice + a_diff) * (1 + thermal_lat_diff)

    return xlattice_final


# Load old experimental data
T1_old, lat1_old = np.loadtxt("data/Francisco_h1.csv", unpack=True)
T1_old = T1_old[:-1]
lat1_old = lat1_old[:-1]
T2_old, lat2_old = np.loadtxt("data/Francisco_h2.csv", unpack=True)
offset = 0.0018
lat2_old = lat2_old - offset

# Use the same transformation functions but with old data
xlattice_sample_12_old = transform_xlattice_old(T_list, TE_list, xlattice)
A_coeff_old, B_coeff_old, C_coeff_old, shrink_parameter_C_old, small_misfit_D_old = obtain_fitting_parameters(
    xlattice_sample_12_old
)

# Compute alpha for old datasets
alpha1_old = (C_coeff_old + B_coeff_old * T1_old + A_coeff_old * T1_old**2 + small_misfit_D_old - lat1_old) / (
    -shrink_parameter_C_old
)
alpha2_old = (C_coeff_old + B_coeff_old * T2_old + A_coeff_old * T2_old**2 + small_misfit_D_old - lat2_old) / (
    -shrink_parameter_C_old
)

# Find transition temperatures for old data
# breakpoint()
T_trans1_old, idx_trans1_old = find_transition_temperature(T1_old, alpha1_old, min_idx=30, max_idx=150)
T_trans2_old, idx_trans2_old = find_transition_temperature(T2_old, alpha2_old, min_idx=30, max_idx=150)
print(f"Transition temperature for Aged state: {T_trans1_old}K")
print(f"Transition temperature for Quenched state: {T_trans2_old}K")

# Plot old datasets
fig_old, ax_old = plt.subplots(figsize=(2.8, 2.6))

# Plot Francisco h1 (1st heating)
ax_old.plot(T1_old[1:idx_trans1_old], alpha1_old[1:idx_trans1_old], "-", c="#216b7b", alpha=1.0, linewidth=1.5)
ax_old.plot(
    T1_old[idx_trans1_old - 1 :],
    alpha1_old[idx_trans1_old - 1 :],
    "-",
    c="#216b7b",
    alpha=1,
    linewidth=1.5,
    label="Aged state",
)
# ax_old.vlines(
#     T_trans1_old,
#     ymin=0,
#     ymax=alpha1_old[idx_trans1_old],
#     color="#216b7b",
#     linestyles="--",
#     alpha=0.5,
#     linewidth=0.8,
#     zorder=0,
# )

# Plot Francisco h2 (2nd heating)
ax_old.plot(T2_old[1:idx_trans2_old], alpha2_old[1:idx_trans2_old], "-", c="#9ed925", alpha=1.0, linewidth=1.5)
ax_old.plot(
    T2_old[idx_trans2_old - 1 :],
    alpha2_old[idx_trans2_old - 1 :],
    "-",
    c="#9ed925",
    alpha=1,
    linewidth=1.5,
    label="Quenched state",
)
# ax_old.vlines(
#     T_trans2_old,
#     ymin=0,
#     ymax=alpha2_old[idx_trans2_old],
#     color="#9ed925",
#     linestyles="--",
#     alpha=0.5,
#     linewidth=0.8,
#     zorder=0,
# )

ax_old.plot(T_fit, WC_spline(T_fit), "-", c="#6B93E4", label="Equilibrium CSRO", linewidth=1.5, zorder=0)

ax_old.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax_old.set_xlabel("Temperature (K)", fontsize=8)
ax_old.legend(loc="best")
ax_old.spines["right"].set_visible(True)
ax_old.spines["top"].set_visible(True)
ax_old.set_xlim([475, 1225])
ax_old.set_ylim([0.34, 0.95])
ax_old.yaxis.set_label_coords(-0.10, 0.5)
ax_old.xaxis.set_label_coords(0.5, -0.08)
fig_old.savefig("figures/effective_alpha_old.png", dpi=300, transparent=False)

# # Compute effective temperatures for old datasets
# T_eff1_old = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha1_old])
# T_eff2_old = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha2_old])

# fig_eff_old, ax_eff_old = plt.subplots(figsize=(2.8, 2.6))
# ax_eff_old.plot(T1_old, T_eff1_old, "-", c="#216b7b", linewidth=1.5, label="Francisco h1")
# ax_eff_old.plot(T2_old, T_eff2_old, "-", c="#37b35f", linewidth=1.5, label="Francisco h2")
# ax_eff_old.plot([400, 1200], [400, 1200], "--", c="gray", alpha=0.5, label="Equilibrium")
# ax_eff_old.set_xlabel("Temperature (K)", fontsize=8)
# ax_eff_old.set_ylabel("Effective Temperature (K)", fontsize=8)
# ax_eff_old.legend(loc="best")
# ax_eff_old.spines["right"].set_visible(True)
# ax_eff_old.spines["top"].set_visible(True)
# ax_eff_old.yaxis.set_label_coords(-0.15, 0.5)
# ax_eff_old.xaxis.set_label_coords(0.5, -0.08)
# fig_eff_old.savefig("figures/effective_temperature_old.png", dpi=300, transparent=False)
