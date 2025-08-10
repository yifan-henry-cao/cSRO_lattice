import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

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
T_fitting = np.arange(300, 1901, 100)
WC_fitting = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, :, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K
WC_spline = UnivariateSpline(T_fitting, WC_fitting, s=None)
T_fit = np.linspace(300, 1900, 101)
# Calculate the derivative of CSRO amount
spline_derivative = WC_spline.derivative()
dCSRO_dT = spline_derivative(T_fit)

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

# Fit quadratic function for random lattice
coeffs, residuals, _, _, _ = np.polyfit(TE_list, lattice_random, deg=2, full=True)
A_coeff, B_coeff, C_coeff = coeffs
print(f"Quadratic fit: {residuals}")

random_estimate = lambda T: 2 * A_coeff * T + B_coeff

# Compute CSRO contribution to thermal expansion
lattice_fitted = (
    A_coeff * T_fit**2 + B_coeff * T_fit + C_coeff + shrink_parameter_C * WC_spline(T_fit)
)  # For normalization of CTE
CTE_CSRO = shrink_parameter_C * dCSRO_dT
CTE_random = random_estimate(T_fit)
total_CTE = CTE_random + CTE_CSRO

# Compute the ratio of CSRO conribution to thermal expansion
ratio_CSRO = CTE_CSRO / total_CTE

################################################################################
# Plot thermal expansion evolution.                                            #
################################################################################
os.makedirs("figures/process_figures/", exist_ok=True)
T_avg, sim_CTE = compute_CTE(TE_list, xlattice_transformed)
average_CTE = np.mean(sim_CTE, axis=0)
# Fit linear regression
coeffs = np.polyfit(T_avg, average_CTE, deg=1)
slope_average, y_intercept_average = coeffs

# Fit linear regression
T_random, CTE_random = compute_CTE_nonuniform(TE_list, lattice_random)
coeffs = np.polyfit(T_random, CTE_random, deg=1)
slope_random, y_intercept_random = coeffs

# Start figure.
fig, ax = plt.subplots(figsize=(3.5 * 1.3, 2.69))

# Plot.
for idx, T0 in enumerate(T_list):
    ax.plot(T_avg, sim_CTE[idx, :], "-o", c=cmap.to_rgba(T0))
ax.plot(T_avg, average_CTE, "o", c="black", label="Average")
ax.plot(T_avg, slope_average * T_avg + y_intercept_average, "-", c="black", label="Linear fit")
ax.plot(T_random, CTE_random, "o", c="orange", label="Random CTE")
ax.plot(T_random, slope_random * T_random + y_intercept_random, "-", c="orange")

# Add colorbar
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

# Add details.
ax.set_ylabel(r"Thermal expansion coefficient (10$^{-6}$ K$^{-1}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
# ax.yaxis.set_label_coords(-0.08, 0.5)
# ax.set_xlim(290, 1310)
ax.set_ylim(0, 30)
ax.legend(loc="lower left", fontsize=7)

# Add text with equation
equation_text = rf"TEC = ${y_intercept_average:.2f}$ + $T\times${slope_average:.2E} $\mathring{{\mathrm{{A}}}}$"
equation_text_random = rf"TEC = ${y_intercept_random:.2f}$ + $T\times${slope_random:.2E} $\mathring{{\mathrm{{A}}}}$"
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=8, verticalalignment="top")
plt.text(0.05, 0.90, equation_text_random, transform=plt.gca().transAxes, fontsize=8, verticalalignment="top")

# Save figure.
fig.savefig("./figures/process_figures/thermal_expansion_test.png", dpi=300, transparent=False)
plt.close()

################################################################################
# Plot lattice vs WC for all TE                                               #
################################################################################

# Start figure.
fig, ax = plt.subplots(figsize=(3.5 * 1.0, 2.69))

# Fit linear regression
coeffs, residuals, _, _, _ = np.polyfit(TE_list, lattice_random, deg=1, full=True)
slope, y_intercept = coeffs
print(f"Linear fit: {residuals}")

# Fit linear regression
coeffs, residuals, _, _, _ = np.polyfit(TE_list, lattice_random, deg=2, full=True)
A_coeff, B_coeff, C_coeff = coeffs
print(f"Quadratic fit: {residuals}")

# Plot for each TE temperature
# ax.plot(TE_list, lattice_random, "-", color="black", linewidth=1, zorder=1)
for idx, TE in enumerate(TE_list):
    ax.plot(TE, lattice_random[idx], "o", color=cmap.to_rgba(TE), markersize=3, zorder=0)
ax.plot(TE_list, slope * TE_list + y_intercept, "-", color="black", linewidth=1, zorder=1, label="Linear fit")
ax.plot(
    TE_list,
    A_coeff * TE_list**2 + B_coeff * TE_list + C_coeff,
    "-",
    color="orange",
    linewidth=1,
    zorder=1,
    label="Quadratic fit",
)

# Add text with equation
equation_text = rf"$a^{{0}}(T)$ = {y_intercept:.2f} + $T\times${slope:.2E} $\mathring{{\mathrm{{A}}}}$"
equation_text_quadratic = (
    rf"$a^{{0}}(T)$ = {C_coeff:.2f} + $T\times${B_coeff:.2E} + $T^2\times${A_coeff:.2E} $\mathring{{\mathrm{{A}}}}$"
)
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=6, verticalalignment="top")
plt.text(0.05, 0.90, equation_text_quadratic, transform=plt.gca().transAxes, fontsize=6, verticalalignment="top")

# Add details
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Lattice parameter baseline $a^{0}$ ($\mathring{\mathrm{A}}$)", fontsize=8)

# Save figure.
fig.savefig("./figures/process_figures/lattice_baseline_test.png", dpi=300, transparent=False)
plt.close()

################################################################################
# Plot lattice temperature evolution.                                          #
################################################################################

# Start figure.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5 * 1.6, 2.69))

# Plot experimental data in first subplot.
ax1.plot(T_fitting, WC_fitting, "o", c="#216b7b")
ax1.plot(T_fit, WC_spline(T_fit), "-", c="#216b7b", label="Equilibrium CSRO")

# Add details for first subplot.
ax1.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax1.set_xlabel("Temperature (K)", fontsize=8)
ax1.legend(loc="best")

# Calculate and plot derivatives in second subplot
spline_derivative = WC_spline.derivative()
derivative_values = spline_derivative(T_fit)

# Calculate numerical derivative of original data
T_centers = (T_fitting[1:] + T_fitting[:-1]) / 2
numerical_derivative = np.diff(WC_fitting) / np.diff(T_fitting)

ax2.plot(T_centers, numerical_derivative, "o", c="#216b7b", label="Data")
ax2.plot(T_fit, derivative_values, "-", c="#216b7b", label="Spline fit")

# Add details for second subplot.
ax2.set_ylabel(r"d$\alpha^{total}$/dT (K$^{-1}$)", fontsize=8)
ax2.set_xlabel("Temperature (K)", fontsize=8)
ax2.legend(loc="best")

# Adjust layout
plt.tight_layout()

# Save figure.
fig.savefig("./figures/process_figures/alpha_vs_T_test.png", dpi=300, transparent=False)


################################################################################
# Plot CTE evolution.                                                          #
################################################################################
def convolve_1D(arr, window_size=30):
    kernel = np.ones(window_size) / window_size
    averaged_array = np.convolve(arr, kernel, mode="valid")
    return averaged_array


# Start figure.
fig, ax = plt.subplots(figsize=(3.5 * 0.8, 2.69))

ax.plot(T_fit, total_CTE / lattice_fitted * 1e6, "-", color="#216b7b", label="Equilbirum")
ax.plot(T_fit, (total_CTE - CTE_CSRO) / lattice_fitted * 1e6, "-", color="#55be0e", label="Random")
ax.set_ylim(10.5, 24)
ax.set_xlim(400, 1200)

T1, lat1 = np.loadtxt("data/Francisco_h1.csv", unpack=True)
# T1_avg, CTE1 = compute_CTE(T1, lat1)
T1_avg, CTE1 = compute_CTE_nonuniform(T1[:-1], lat1[:-1])
T1_avg = convolve_1D(T1_avg)
CTE1 = convolve_1D(CTE1)

T2, lat2 = np.loadtxt("data/Francisco_h2.csv", unpack=True)
# print(np.mean(T2[1:]-T2[:-1]))
T2_avg, CTE2 = compute_CTE_nonuniform(T2, lat2)
T2_avg = convolve_1D(T2_avg)
CTE2 = convolve_1D(CTE2)

ax.plot(T1_avg, CTE1, "o", c="#216b7b", alpha=0.5, label="Aged state")
ax.plot(T2_avg, CTE2, "o", c="#55be0e", alpha=0.5, label="Quenched state")

# Add details.
ax.set_ylabel(r"TEC ($10^{-6}\,K^{-1}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.09, 0.5)
ax.xaxis.set_label_coords(0.5, -0.07)

# Save figure.
fig.savefig("./figures/process_figures/thermal_expansion_fitted_test.png", dpi=300, transparent=False)
plt.close()


################################################################################
# Plot CSRO contribution.                                                      #
################################################################################
def convolve_1D(arr, window_size=30):
    kernel = np.ones(window_size) / window_size
    averaged_array = np.convolve(arr, kernel, mode="valid")
    return averaged_array


# Start figure.
fig, ax = plt.subplots(figsize=(2.8, 2.6))

ax.plot(T_fit, ratio_CSRO * 100, "-", color="#6B93E4", linewidth=1.5, label="Equilbirum")
# ax.set_ylim(0, 100)
ax.set_xlim(400, 1500)

T1, lat1 = np.loadtxt("data/Francisco_h1.csv", unpack=True)
T1_avg, CTE1 = compute_CTE_nonuniform(T1[:-1], lat1[:-1])
T1_avg = convolve_1D(T1_avg)
CTE1 = convolve_1D(CTE1)
lat1_avg = convolve_1D(lat1[1:-1])
ratio_CSRO1 = (CTE1 - random_estimate(T1_avg) / lat1_avg * 1e6) / CTE1 * 100

T2, lat2 = np.loadtxt("data/Francisco_h2.csv", unpack=True)
T2_avg, CTE2 = compute_CTE_nonuniform(T2, lat2)
T2_avg = convolve_1D(T2_avg)
CTE2 = convolve_1D(CTE2)
lat2_avg = convolve_1D(lat2[1:])
ratio_CSRO2 = (CTE2 - random_estimate(T2_avg) / lat2_avg * 1e6) / CTE2 * 100

ax.plot(T1_avg, ratio_CSRO1, "-", c="#216b7b", alpha=0.9, label="Aged state")
ax.plot(T2_avg, ratio_CSRO2, "-", c="#55be0e", alpha=0.9, label="Quenched state")

ax.axhline(0, color="black", alpha=0.5, linestyle="--", zorder=0)

# Add details.
ax.set_ylabel(r"CSRO contribution to TEC (%)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.09, 0.5)
ax.xaxis.set_label_coords(0.5, -0.07)

# Save figure.
fig.savefig("./figures/CSRO_contribution.png", dpi=300, transparent=False)
plt.close()
