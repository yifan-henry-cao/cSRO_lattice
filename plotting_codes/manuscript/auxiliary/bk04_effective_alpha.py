import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

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


def transform_xlattice(T_list, TE_list, xlattice):
    # Load experimental data
    T1, lat1 = np.loadtxt("data/Francisco_h1.csv", unpack=True)
    T2, lat2 = np.loadtxt("data/Francisco_h2.csv", unpack=True)

    # Get reference points
    a0_exp = lat1[-3]  # Lattice at 1071K
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)
    a0_sim = initial_spline(1071, 1071)
    a_diff = a0_exp - a0_sim

    T1_avg, CTE1 = compute_CTE_nonuniform(T1[:44], lat1[:44])
    T2_avg, CTE2 = compute_CTE_nonuniform(T2[:44], lat2[:44])
    T_avg, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)

    thermal_exp = (np.mean(CTE1) + np.mean(CTE2)) / 2
    thermal_sim = np.mean(sim_CTE, axis=0)[7:13].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6

    # Apply thermal correction to adjusted lattice
    thermal_lat_diff = thermal_diff * (TE_list - 1075)  # Intercept at 1075K
    xlattice_final = (xlattice + a_diff) * (1 + thermal_lat_diff)

    # print(f"a_diff: {a_diff[0][0]}")
    # print(f"thermal_diff: {thermal_diff}")
    return xlattice_final


xlattice_transformed = transform_xlattice(T_list, TE_list, xlattice)

################################################################################
# First get the baseline correction from x01                                   #
################################################################################

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

# Perform linear fit to express baseline
coeffs = np.polyfit(TE_list, lattice_random, deg=1)
random_expansion_B, lattice_0K_A = coeffs

################################################################################
# Load experimental lattice data from x01                                      #
################################################################################
# Load experimental data
T1, lat1 = np.loadtxt("data/Francisco_h1.csv", unpack=True)
T1 = T1[:-1]
lat1 = lat1[:-1]
T2, lat2 = np.loadtxt("data/Francisco_h2.csv", unpack=True)
offset = 0.0018
lat2 = lat2 - offset
print(f"2nd heating offset value: {offset}A")

alpha1 = (lattice_0K_A + random_expansion_B * T1 + small_misfit_D - lat1) / (-shrink_parameter_C)
alpha2 = (lattice_0K_A + random_expansion_B * T2 + small_misfit_D - lat2) / (-shrink_parameter_C)
# print(f"lattice_0K_A: {lattice_0K_A}")
# print(f"random_expansion_B: {random_expansion_B}")
# print(f"small_misfit_D: {small_misfit_D}")
# print(f"shrink_parameter_C: {shrink_parameter_C}")

# Start figure.
fig, ax = plt.subplots(figsize=(2.8, 2.6))

# Plot experimental data.
ax.plot(T1, alpha1, "-", c="#216b7b", linewidth=1.5, label="Aged state")
ax.plot(T2, alpha2, "-", c="#9ed925", linewidth=1.5, label="Quenched state")
ax.plot(T_list[:-4], WC_1nn_sum[:-4], "o", c="#6B93E4", zorder=0)
ax.plot(T_fit, WC_spline(T_fit), "-", c="#6B93E4", label="Equilibrium CSRO", zorder=0)
# ax.plot(T_list, WC_1nn_sum, "k--", linewidth=1)

# Add details.
ax.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
# ax.set_xlim(0, 2000)
ax.legend(loc="lower left")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig("figures/effective_alpha.png", dpi=300, transparent=False)


################################################################################
# Compute effective temperatures                                               #
################################################################################
# Compute effective temperatures for both curves
T_search = np.linspace(400, 1600, 1200)
WC_values = WC_spline(T_search)
T_eff1 = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha1])
T_eff2 = np.array([T_search[np.abs(WC_values - a).argmin()] for a in alpha2])

# Create new figure for effective temperatures
fig_eff, ax_eff = plt.subplots(figsize=(2.8, 2.6))

# Plot T_effective vs T_actual
ax_eff.plot(T1, T_eff1, "-", c="#216b7b", linewidth=1.5, label="Aged state")
ax_eff.plot(T2, T_eff2, "-", c="#9ed925", linewidth=1.5, label="Quenched state")
ax_eff.plot([400, 1200], [400, 1200], "--", c="gray", alpha=0.5, label="Equilibrium")

ax_eff.set_xlabel("Temperature (K)", fontsize=8)
ax_eff.set_ylabel("Effective Temperature (K)", fontsize=8)
ax_eff.legend(loc="best")
ax_eff.spines["right"].set_visible(True)
ax_eff.spines["top"].set_visible(True)
ax_eff.yaxis.set_label_coords(-0.15, 0.5)
ax_eff.xaxis.set_label_coords(0.5, -0.08)

# Save figure
fig_eff.savefig("figures/effective_temperature.png", dpi=300, transparent=False)
