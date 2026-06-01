import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os
from scipy.signal import savgol_filter

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
WC_1nn_sum = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, 1:-3, :]), axis=-1
)  # (3, 17, 6) -> (13,) 1st neighbor, 400->1600K
T_fitting = np.arange(300, 1901, 100)
WC_fitting = np.sum(
    np.abs(np.load("data/WC_params/WC_avg.npy")[0, :, :]), axis=-1
)  # (3, 17, 6) -> (17,) 1st neighbor, 300->1900K
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


# --- Load experimental volumetric data and transform to lattice constant ---
def load_volumetric_lattice():
    data_dir = "data"
    files = [
        "Vol_vs_T_h1.csv",
        "Vol_vs_T_h2.csv",
        "Vol_vs_T_h3.csv",
    ]
    labels = ["Aged state", "Re-heated state", "Quenched state"]
    # Colorblind-friendly: blue, green, yellow-green
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
    T_exp_all, a_exp_all, _, _ = load_volumetric_lattice()
    T1, lat1 = T_exp_all[0], a_exp_all[0]
    T2, lat2 = T_exp_all[1], a_exp_all[1]
    T3, lat3 = T_exp_all[2], a_exp_all[2]
    a0_exp = lat3[-1]  # Lattice at 1073.19K
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)
    a0_sim = initial_spline(1073.19, 1073.19)
    a_diff = a0_exp - a0_sim
    T1_avg, CTE1 = compute_CTE_windowed(T1[:1200], lat1[:1200])
    T2_avg, CTE2 = compute_CTE_windowed(T2[:1200], lat2[:1200])
    T3_avg, CTE3 = compute_CTE_windowed(T3[:1200], lat3[:1200])
    T_avg, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)
    thermal_exp = (np.mean(CTE1[15:-15]) + np.mean(CTE2[15:-15]) + np.mean(CTE3[15:-15])) / 3
    thermal_sim = np.mean(sim_CTE, axis=0)[7:15].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6
    thermal_lat_diff = thermal_diff * (TE_list - 1075)  # Intercept at 1075K
    xlattice_final = (xlattice + a_diff) * (1 + thermal_lat_diff)
    return xlattice_final


# --- Use new experimental data for all further analysis ---
T_exp_all, a_exp_all, exp_labels, exp_colors = load_volumetric_lattice()
T1, lat1 = T_exp_all[0], a_exp_all[0]
T2, lat2 = T_exp_all[1], a_exp_all[1]
T3, lat3 = T_exp_all[2], a_exp_all[2]

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

# Compute the ratio of CSRO contribution to thermal expansion
ratio_CSRO = CTE_CSRO / total_CTE

################################################################################
# Plot CSRO contribution.                                                      #
################################################################################
def convolve_1D(arr, window_size=30):
    kernel = np.ones(window_size) / window_size
    averaged_array = np.convolve(arr, kernel, mode="valid")
    return averaged_array

window_size = 31
savgol_win = 101 if window_size < 51 else window_size | 1  # odd window, >= window_size
polyorder = 3

# Start figure.
fig, ax = plt.subplots(figsize=(2.8, 2.6))

ax.plot(T_fit, ratio_CSRO * 100, "-", color="#6B93E4", linewidth=1, label="Equilbirum")
ax.set_xlim(400, 1500)

for T_exp, lat_exp, color, label in zip([T1, T2, T3], [lat1, lat2, lat3], exp_colors, exp_labels):
    T_avg, CTE = compute_CTE_windowed(T_exp, lat_exp, window_size=window_size)
    margin = window_size // 2
    valid = ~np.isnan(CTE)
    valid[:margin] = False
    valid[-margin:] = False
    if np.sum(valid) > polyorder:
        CTE_smooth = convolve_1D(CTE[valid], window_size=savgol_win)
        T_avg = convolve_1D(T_avg[valid], window_size=savgol_win)
    else:
        CTE_smooth = CTE
    lat_avg = np.interp(T_avg, T_exp, lat_exp)
    ratio_CSRO_exp = (CTE_smooth - random_estimate(T_avg) / lat_avg * 1e6) / CTE_smooth * 100
    ax.plot(T_avg, ratio_CSRO_exp, "-", c=color, alpha=0.9, label=label, linewidth=1.5)

ax.axhline(0, color="black", alpha=0.5, linestyle="--", zorder=0)

# Add details.
ax.set_ylabel(r"CSRO contribution to TEC (%)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best")
ax.set_xlim([525, 1275])
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.12, 0.5)
ax.xaxis.set_label_coords(0.5, -0.07)

# Save figure.
fig.savefig("./figures/CSRO_contribution.pdf")
plt.close()
