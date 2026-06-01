import os
import json

import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from scipy.interpolate import RectBivariateSpline

plt.style.use("paper")

from scipy.interpolate import CubicSpline


################################################################################
# Load and process simulation data (same setup as 04_effective_alpha.py)       #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)

# Load lattice and SRO data
xlattice_data = np.load("data/sim_data/lattice.npy")  # (13, 33, 10)
xlattice = np.mean(xlattice_data, axis=-1)  # (13, 33, 10) -> (13, 33)
WC_1nn_sum = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K
WC_spline = CubicSpline(T_list, WC_1nn_sum)
T_fit = np.linspace(400, 1200, 101)

os.makedirs("figures", exist_ok=True)


################################################################################
# Helper functions (copied from 04_effective_alpha.py)                         #
################################################################################
def compute_CTE_nonuniform(T_list, xlattice):
    # Simple finite difference for CTE with non-uniform temperature spacing
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


def transform_xlattice_old(T_list, TE_list, xlattice):
    """Apply the old Francisco-datasets-based thermal expansion correction."""
    # Load old experimental data
    T1_old_full, lat1_old_full = np.loadtxt("data/Francisco_h1.csv", unpack=True)
    T2_old_full, lat2_old_full = np.loadtxt("data/Francisco_h2.csv", unpack=True)

    # Get reference points using old data
    a0_exp = lat2_old_full[-1] - 0.0018  # Use h2 as reference, minus exp offset
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)
    a0_sim = initial_spline(1073.19, 1073.19)
    a_diff = a0_exp - a0_sim

    # Compute CTE for old data (use the same window as in 04_effective_alpha.py)
    T1_avg, CTE1 = compute_CTE_nonuniform(T1_old_full[:50], lat1_old_full[:50])
    T2_avg, CTE2 = compute_CTE_nonuniform(T2_old_full[:50], lat2_old_full[:50])
    T_avg, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)

    thermal_exp = (np.mean(CTE1[1:]) + np.mean(CTE2)) / 2
    thermal_sim = np.mean(sim_CTE, axis=0)[7:13].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6
    print(f"Old data thermal_diff: {thermal_diff}")

    # Apply thermal correction to adjusted lattice
    thermal_lat_diff = thermal_diff * (TE_list - 1073.19)  # Intercept at 1073K
    xlattice_final = (xlattice + a_diff) * (1 + thermal_lat_diff)

    return xlattice_final


def obtain_fitting_parameters(xlattice_transformed):
    """Obtain polynomial baseline and WC-coupling parameters."""
    lattice_random = np.zeros_like(TE_list).astype(np.float64)
    lattice_adjusted = np.zeros_like(xlattice)

    for idx, TE in enumerate(TE_list):
        lattice_at_TE = xlattice_transformed[:, idx]

        # Fit linear regression vs WC_1nn_sum
        coeffs = np.polyfit(WC_1nn_sum, lattice_at_TE, deg=1)
        slope, y_intercept = coeffs

        lattice_random[idx] = y_intercept

        # Subtract y-intercept and get adjusted lattice
        adjusted_lattice = lattice_at_TE - y_intercept
        lattice_adjusted[:, idx] = adjusted_lattice

    # Adjusted lattice mean to obtain constant c
    lattice_adjusted_mean = np.mean(lattice_adjusted, axis=1)

    # Fit linear regression vs WC_1nn_sum
    coeffs = np.polyfit(WC_1nn_sum, lattice_adjusted_mean, deg=1)
    shrink_parameter_C, small_misfit_D = coeffs

    # Fit quadratic function for random lattice vs TE_list
    coeffs_quad, residuals, _, _, _ = np.polyfit(
        TE_list, lattice_random, deg=2, full=True
    )
    A_coeff, B_coeff, C_coeff = coeffs_quad
    fitting_coeffs = [A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D]

    return fitting_coeffs


################################################################################
# Process old Francisco datasets                                               #
################################################################################
# Load old experimental data (matching the logic after line 330 in 04_effective_alpha.py)
T1_old, lat1_old = np.loadtxt("data/Francisco_h1.csv", unpack=True)
T1_old = T1_old[:-1]
lat1_old = lat1_old[:-1]
T2_old, lat2_old = np.loadtxt("data/Francisco_h2.csv", unpack=True)
offset = 0.0018
lat2_old = lat2_old - offset

# Use the same transformation function with old data
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


################################################################################
# Plot: effective_alpha_old                                                    #
################################################################################
fig, ax = plt.subplots(figsize=(2.8, 2.6))

# Plot Francisco h1 (1st heating, aged state)
ax.plot(T1_old, alpha1_old, "-", c="#216b7b", alpha=1.0, linewidth=1.5, label="Aged state")

# Plot Francisco h2 (2nd heating, quenched state)
ax.plot(T2_old, alpha2_old, "-", c="#9ed925", alpha=1.0, linewidth=1.5, label="Quenched state")

# Equilibrium CSRO reference
ax.plot(T_fit, WC_spline(T_fit), "-", c="#6B93E4", label="Equilibrium CSRO", linewidth=1.5, zorder=0)

ax.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_xlim([475, 1225])
ax.set_ylim([0.34, 0.95])
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

fig.savefig("figures/effective_alpha.pdf")
plt.close()
