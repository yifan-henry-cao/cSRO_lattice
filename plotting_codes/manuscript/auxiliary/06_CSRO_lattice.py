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


def convert_WC_to_pij(WC_array):
    multiplier = np.array([1, 2, 2, 1, 2, 1])  # 1 for i=j, 2 for i!=j
    pij = multiplier * (1 - WC_array) / 9  # ci=cj=1/3
    return pij


# %% This section computes the relative contribution between bond length changes and bond distribution changes (frac_BL)
# WC_avg = np.load("data/WC_params/WC_avg.npy")[0, 1:-7, :]  # (3, 17, 6) -> (9 ,6) 400K -> 1200K CSRO
WC_avg = np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]  # (3, 17, 6) -> (13 ,6) 400K -> 1600K CSRO
p0 = convert_WC_to_pij(np.zeros(6))  # (6,)
p_SRO = convert_WC_to_pij(WC_avg)
dp = p_SRO - p0  # (9, 6)

upper_indices = np.triu_indices(3)
b0 = np.mean(np.load("data/sim_data/bond_length_TE_RSS.npy"), axis=1)[
    24, upper_indices[0], upper_indices[1]
]  # (33, 40, 3, 3) -> (6) at 1000K
# b0 = np.mean(b0_raw, axis=0)  # (6,)
bond_length_TSRO = np.load("data/sim_data/bond_length_TSRO.npy")  # (T, ID, 3, 3), thermal temp at 1000K
# bond_mean_TSRO = np.mean(bond_length_TSRO, axis=1)[:-4, :, :]  # (13, 3, 3) -> (9, 3, 3)
bond_mean_TSRO = np.mean(bond_length_TSRO, axis=1)  # (13, 3, 3)
b_TSRO = bond_mean_TSRO[:, upper_indices[0], upper_indices[1]]  # shape: (9, 6)
db = b_TSRO - b0  # Shape (9, 6)

# delta_b_bar = np.sum(p_SRO * b_TSRO, axis=1) - np.sum(p0 * b0)  # (9,)
b_dp = np.sum(b0 * dp, axis=1)  # (9,)
p_db = np.sum(p0 * db, axis=1)
dpdb = np.sum(dp * db, axis=1)
frac_WC = (b_dp + dpdb / 2) / (b_dp + p_db + dpdb)
frac_BL = (p_db + dpdb / 2) / (b_dp + p_db + dpdb)
frac_BL_spline = CubicSpline(T_list, frac_BL)

# print(dpdb / (b_dp + p_db) * 100)
# %% Lattice difference
A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D = obtain_fitting_parameters(xlattice)
# lat_ref = xlattice[-1, -1]  # 1200K with 1600K CSRO
lat_ref = A_coeff * 1660**2 + B_coeff * 1660 + C_coeff + small_misfit_D
T_fitted = np.linspace(400, 1200, 101)
WC_fitted = WC_spline(T_fitted)
BL_fitted = frac_BL_spline(T_fitted)
# lat_thermal_fitted = (
#     A_coeff * T_fitted**2 + B_coeff * T_fitted + C_coeff + shrink_parameter_C * WC_1nn_sum[-1] + small_misfit_D
# )
lat_thermal_fitted = A_coeff * T_fitted**2 + B_coeff * T_fitted + C_coeff + small_misfit_D
lat_eq_fitted = A_coeff * T_fitted**2 + B_coeff * T_fitted + C_coeff + shrink_parameter_C * WC_fitted + small_misfit_D

fig, ax = plt.subplots(figsize=(3.5, 3))
ax.plot(T_fitted, WC_fitted, label="WC", color="black", linestyle="--")
ax.plot(T_list, WC_1nn_sum, "ro")
fig.savefig("WC_spline.png", dpi=300, transparent=False)

fig, ax = plt.subplots(figsize=(3.5, 3))
ax.plot(T_fitted, BL_fitted, label="r_BL", color="black", linestyle="--")
ax.plot(T_list, frac_BL, "ro")
fig.savefig("BL_spline.png", dpi=300, transparent=False)

dA_thermal = lat_ref - lat_thermal_fitted
dA_BL = (lat_thermal_fitted - lat_eq_fitted) * BL_fitted + dA_thermal  # Estimate dA with only BL contribution
dA_CSRO = lat_ref - lat_eq_fitted
CSRO_contribution = (dA_CSRO - dA_thermal) / dA_CSRO * 100
# print((dA_CSRO - dA_thermal) / dA_CSRO)
# print(T_fitted)

# Plot example
fig, ax = plt.subplots(figsize=(3.5 * 1.05, 2.69))
# Plot the three lines
ax.plot(T_fitted, dA_thermal, label="Random Solid Solution", color="black", linestyle="--")
ax.plot(T_fitted, dA_CSRO, label="Equilibrium CSRO", color="black")

# Fill between curves
# "lightcoral"
ax.fill_between(
    T_fitted,
    dA_thermal,
    np.zeros_like(dA_thermal),
    label="Thermal contribution",
    color=thermal_color,
    alpha=0.5,
    edgecolor="none",
)
# "gold"
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
# ax.set_title("Lattice parameter change")
ax.legend(loc="lower left", frameon=True, framealpha=0.8, fontsize=6)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
# ax.set_ylim(0, 0.054)
ax.set_ylim(0, 0.094)
ax.set_xlim(400 - 1, 1200)
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)
# ax.grid(True)
# plt.tight_layout()
fig.savefig("figures/CSRO_lattice.png", dpi=300, transparent=False)
plt.close()

fig, ax = plt.subplots(figsize=(3.5 * 0.9, 2.5))
# ax.plot(T_list, frac_BL, color="black", alpha=0.8, linestyle="--", label="BL fraction")
# ax.plot(T_list, frac_BL, "o", color="black")
mean_BL = np.mean(frac_BL)
print(f"averge contribution from bond length: {mean_BL}")
# ax.axhline(y=mean_BL, color="black", alpha=0.8, linestyle="--")
ax.fill_between(T_list, 0, frac_BL, color=bondlength_color, alpha=0.9, label="Bond length contribution")
ax.fill_between(T_list, frac_BL, 1, color=bonddist_color, alpha=1.0, label="CSRO contribution")
ax.set_xlabel("Temperature (K)", fontsize=12)
# ax.set_ylabel(r"$r_{bond length}$", fontsize=12)
ax.set_ylabel("CSRO contribution", fontsize=12)
ax.set_ylim(0, 1)
ax.set_xlim(400 - 1, 1600)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.tick_params(axis="both", which="major", labelsize=10)
# ax.set_title("Fraction of WC vs Temperature")
# ax.legend()
# plt.tight_layout()
fig.savefig("figures/frac_WC_BL_vs_T.png", dpi=300, transparent=True)
plt.close()

# # Temperature color map
# # Color gradient change, KL divergence? Time consuming, no need if necessary
# # Figure 4d, matched experimental order parameter (mapped from lattice constant)
# # Figure 4e, Sigma|\alpha| vs Temperature for CrCoNi
# # Cr-Cr vs Cr-Co bond length
# # RedtoBlue = LinearSegmentedColormap.from_list("RedtoBlue", ["#1f77b4", "#d62728"])
# Viridis = plt.colormaps["viridis"].reversed()
# # cmap = ScalarMappable(Normalize(min(T_list), max(T_list)), RedtoBlue)
# cmap = ScalarMappable(Normalize(min(WC_1nn_CrCr), max(WC_1nn_CrCr)), Viridis)

# os.system("mkdir -p figures/")

# ################################################################################
# # Plot lattice temperature evolution.                                          #
# ################################################################################

# # Start figure.
# # fig, ax = plt.subplots(figsize=(3.5 * 1.575, 2.69))
# fig, ax = plt.subplots(figsize=(2.8 * 1.2, 2.6))

# # Plot.
# for idx, WC in enumerate(WC_1nn_CrCr):
#     ax.plot(TE_list, xlattice[idx, :], "-o", c=cmap.to_rgba(WC))

# # Add colorbar with specific ticks
# tick_values = np.linspace(min(WC_1nn_CrCr), max(WC_1nn_CrCr), 6)  # 6 ticks between min and max
# cbar = fig.colorbar(
#     cmap,
#     ax=ax,
#     pad=0.03,
#     ticks=tick_values,
#     aspect=60,
#     fraction=1,
#     label=r"$\alpha^{Cr-Cr}$",
#     orientation="vertical",
# )
# # cbar.ax.tick_params(labelsize=6)
# cbar.ax.tick_params(labelsize=5)
# # Format tick labels to show 3 decimal places
# cbar.ax.set_yticklabels([f"{x:.2f}" for x in tick_values])

# # Add details.
# ax.set_ylabel(r"Lattice parameter ($\mathring{\mathrm{A}}$)", fontsize=8)
# ax.set_xlabel("Temperature (K)", fontsize=8)
# # ax.set_xlim(0, 2000)
# # ax.legend()
# ax.spines["right"].set_visible(True)
# ax.spines["top"].set_visible(True)

# # Save figure.
# fig.savefig("figures/lattice_summary.png", dpi=300, transparent=False)
# plt.close()

# Plot CSRO contribution vs temperature
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.plot(T_fitted, CSRO_contribution, "-", color="#6E96E5", linewidth=2, label="CSRO contribution")
ax.set_xlabel("Temperature (K)", fontsize=8)
# ax.set_ylabel(r"CSRO contribution to $\alpha$ (%)", fontsize=8)
ax.set_ylabel(r"CSRO contribution to $\Delta a$ (%)", fontsize=8)
ax.grid(True, alpha=0.3)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
# ax.tick_params(axis="both", which="major", labelsize=10)
ax.set_xlim(400, 1100)
# plt.tight_layout()
fig.savefig("figures/CSRO_contribution_vs_T.png", dpi=300, transparent=False)
plt.close()
