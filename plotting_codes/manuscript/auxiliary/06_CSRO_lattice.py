import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy.interpolate import RectBivariateSpline, CubicSpline

thermal_color = "#79ad89"
bondlength_color = "#FF6D6D"
bonddist_color = "royalblue"

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
WC_spline = CubicSpline(T_list, WC_1nn_sum)


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

    # Fit quadratic function for random lattice
    coeffs, residuals, _, _, _ = np.polyfit(TE_list, lattice_random, deg=2, full=True)
    A_coeff, B_coeff, C_coeff = coeffs
    fitting_coeffs = [A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D]

    return fitting_coeffs


def convert_WC_to_pij(WC_array):
    multiplier = np.array([1, 2, 2, 1, 2, 1])  # 1 for i=j, 2 for i!=j
    pij = multiplier * (1 - WC_array) / 9  # ci=cj=1/3
    return pij


# This section computes the relative contribution between bond length changes and bond distribution changes (frac_BL)
WC_avg = np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]  # (3, 17, 6) -> (13, 6) 400K -> 1600K CSRO
p0 = convert_WC_to_pij(np.zeros(6))  # (6,)
p_SRO = convert_WC_to_pij(WC_avg)
dp = p_SRO - p0  # (13, 6)

upper_indices = np.triu_indices(3)
b0 = np.mean(np.load("data/sim_data/bond_length_TE_RSS.npy"), axis=1)[
    24, upper_indices[0], upper_indices[1]
]  # (33, 40, 3, 3) -> (6) at 1000K
bond_length_TSRO = np.load("data/sim_data/bond_length_TSRO.npy")  # (T, ID, 3, 3), thermal temp at 1000K
bond_mean_TSRO = np.mean(bond_length_TSRO, axis=1)  # (13, 3, 3)
b_TSRO = bond_mean_TSRO[:, upper_indices[0], upper_indices[1]]  # shape: (13, 6)
db = b_TSRO - b0  # Shape (13, 6)

b_dp = np.sum(b0 * dp, axis=1)  # (13,)
p_db = np.sum(p0 * db, axis=1)
dpdb = np.sum(dp * db, axis=1)
frac_WC = (b_dp + dpdb / 2) / (b_dp + p_db + dpdb)
frac_BL = (p_db + dpdb / 2) / (b_dp + p_db + dpdb)
frac_BL_spline = CubicSpline(T_list, frac_BL)

# Lattice difference
A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D = obtain_fitting_parameters(xlattice)
lat_ref = A_coeff * 1660**2 + B_coeff * 1660 + C_coeff + small_misfit_D
T_fitted = np.linspace(400, 1200, 101)
WC_fitted = WC_spline(T_fitted)
BL_fitted = frac_BL_spline(T_fitted)
lat_thermal_fitted = A_coeff * T_fitted**2 + B_coeff * T_fitted + C_coeff + small_misfit_D
lat_eq_fitted = A_coeff * T_fitted**2 + B_coeff * T_fitted + C_coeff + shrink_parameter_C * WC_fitted + small_misfit_D

dA_thermal = lat_ref - lat_thermal_fitted
dA_BL = (lat_thermal_fitted - lat_eq_fitted) * BL_fitted + dA_thermal  # Estimate dA with only BL contribution
dA_CSRO = lat_ref - lat_eq_fitted
CSRO_contribution = (dA_CSRO - dA_thermal) / dA_CSRO * 100

# Plot example
fig, ax = plt.subplots(figsize=(3.5 * 1.05, 2.69))
# Plot the three lines
ax.plot(T_fitted, dA_thermal, label="Random Solid Solution", color="black", linestyle="--")
ax.plot(T_fitted, dA_CSRO, label="Equilibrium CSRO", color="black")

# Fill between curves
ax.fill_between(
    T_fitted,
    dA_thermal,
    np.zeros_like(dA_thermal),
    label="Thermal contribution",
    color=thermal_color,
    alpha=0.5,
    edgecolor="none",
)
ax.fill_between(
    T_fitted,
    dA_BL,
    dA_thermal,
    label="Bond length contribution",
    color=bondlength_color,
    alpha=0.9,
    edgecolor="none",
)
ax.fill_between(
    T_fitted,
    dA_CSRO,
    dA_BL,
    label="Bond distribution contribution",
    color=bonddist_color,
    alpha=1.0,
    edgecolor="none",
)

# Labels and legend
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Relative lattice contraction $\Delta a$ ($\AA$)", fontsize=8)
ax.legend(loc="lower left", frameon=True, framealpha=0.8, fontsize=6)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_ylim(0, 0.094)
ax.set_xlim(400 - 1, 1200)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)
fig.savefig("figures/CSRO_lattice.pdf")
plt.close()

fig, ax = plt.subplots(figsize=(3.5 * 0.9, 2.5))
mean_BL = np.mean(frac_BL)
print(f"averge contribution from bond length: {mean_BL}")
ax.fill_between(T_list, 0, frac_BL, color=bondlength_color, alpha=0.9, label="Bond length contribution")
ax.fill_between(T_list, frac_BL, 1, color=bonddist_color, alpha=1.0, label="CSRO contribution")
ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel("CSRO contribution", fontsize=12)
ax.set_ylim(0, 1)
ax.set_xlim(400 - 1, 1600)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.tick_params(axis="both", which="major", labelsize=10)
fig.savefig("figures/frac_WC_BL_vs_T.pdf")
plt.close()

# Plot CSRO contribution vs temperature
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.plot(T_fitted, CSRO_contribution, "-", color="#6E96E5", linewidth=2, label="CSRO contribution")
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"CSRO contribution to $\Delta a$ (%)", fontsize=8)
ax.grid(True, alpha=0.3)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_xlim(400, 1100)
fig.savefig("figures/CSRO_contribution_vs_T.pdf")
plt.close()
