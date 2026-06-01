import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.interpolate import CubicSpline, RectBivariateSpline

plt.style.use("paper")

################################################################################
# Paths and configuration                                                      #
################################################################################

output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
details_dir = output_dir / "details"
details_dir.mkdir(parents=True, exist_ok=True)

wc_data_dir = Path("data/WC_params")
vacancy_data_dir = Path("data/vacancy_jump_heating")
sim_data_dir = Path("data/sim_data")

T_list_ref = np.arange(400, 1601, 100).astype(int)
TE_list_ref = np.arange(400, 1201, 25).astype(int)


def compute_CTE_nonuniform(T_list, xlattice):
    dT = T_list[1:] - T_list[:-1]
    dL = xlattice[1:] - xlattice[:-1]
    L_avg = (xlattice[1:] + xlattice[:-1]) / 2
    T_avg = (T_list[1:] + T_list[:-1]) / 2
    CTE = (dL / dT) / L_avg * 1e6
    return T_avg, CTE


def compute_CTE(T_list, xlattice):
    dT = T_list[1:] - T_list[:-1]
    dL = xlattice[:, 1:] - xlattice[:, :-1]
    L_avg = (xlattice[:, 1:] + xlattice[:, :-1]) / 2
    T_avg = (T_list[1:] + T_list[:-1]) / 2
    CTE = dL / dT / L_avg * 1e6
    return T_avg, CTE


def obtain_fitting_parameters(xlattice_transformed, WC_1nn_sum):
    lattice_random = np.zeros_like(TE_list_ref, dtype=np.float64)
    lattice_adjusted = np.zeros_like(xlattice_transformed)

    for idx, _ in enumerate(TE_list_ref):
        lattice_at_TE = xlattice_transformed[:, idx]
        slope, y_intercept = np.polyfit(WC_1nn_sum, lattice_at_TE, deg=1)
        lattice_random[idx] = y_intercept
        lattice_adjusted[:, idx] = lattice_at_TE - y_intercept

    lattice_adjusted_mean = np.mean(lattice_adjusted, axis=1)
    shrink_parameter_C, small_misfit_D = np.polyfit(WC_1nn_sum, lattice_adjusted_mean, deg=1)
    A_coeff, B_coeff, C_coeff = np.polyfit(TE_list_ref, lattice_random, deg=2)

    return A_coeff, B_coeff, C_coeff, shrink_parameter_C, small_misfit_D


def transform_xlattice_old(T_list, TE_list, xlattice):
    T1_old_raw, lat1_old_raw = np.loadtxt("data/Francisco_h1.csv", unpack=True)
    T2_old_raw, lat2_old_raw = np.loadtxt("data/Francisco_h2.csv", unpack=True)

    a0_exp = lat2_old_raw[-1] - 0.0018
    initial_spline = RectBivariateSpline(T_list, TE_list, xlattice, kx=1, ky=1, s=0)
    a0_sim = initial_spline(1073.19, 1073.19)
    a_diff = a0_exp - a0_sim

    _, CTE1 = compute_CTE_nonuniform(T1_old_raw[:50], lat1_old_raw[:50])
    _, CTE2 = compute_CTE_nonuniform(T2_old_raw[:50], lat2_old_raw[:50])
    _, sim_CTE = compute_CTE(TE_list, xlattice + a_diff)

    thermal_exp = (np.mean(CTE1[1:]) + np.mean(CTE2)) / 2
    thermal_sim = np.mean(sim_CTE, axis=0)[7:13].mean()
    thermal_diff = (thermal_exp - thermal_sim) * 1e-6

    thermal_lat_diff = thermal_diff * (TE_list - 1073.19)
    return (xlattice + a_diff) * (1 + thermal_lat_diff)

################################################################################
# Load equilibrium CSRO reference data                                         #
################################################################################

print("Loading equilibrium CSRO reference data (s07)...")

T_list_ref = np.arange(400, 1801, 100).astype(int)  # 400->1700K
WC_spline = None
T_fit = None

try:
    WC_1nn_sum_ref = np.sum(
        np.abs(np.load(wc_data_dir / "WC_avg.npy")[0, 1:-1, :]),
        axis=-1,
    )  # (14,) 1st neighbor, 400->1700K
    WC_spline = CubicSpline(T_list_ref, WC_1nn_sum_ref)
    T_fit = np.linspace(400, 1800, 131)
    print("  Loaded equilibrium CSRO spline (extended to 1700K)")
except Exception as e:
    print(f"  Warning: Could not load equilibrium CSRO reference: {e}")

################################################################################
# Load vacancy heating ramp simulation data                                   #
################################################################################

print("\nLoading vacancy heating ramp simulation data (s07)...")
alpha_ramp_solid = None
T_ramp_solid = None
alpha_ramp_hold = None
T_ramp_hold = None

_ramp_h1, _ramp_h2, _ramp_rc = 4.0, 80.0, 5.0
_ramp_hold_start = _ramp_h1 + _ramp_h2 + _ramp_rc  # 89 ns

def _vT_ramp(t):
    if t <= _ramp_h1:
        return 400.0 + 400.0 * t / _ramp_h1
    elif t <= _ramp_h1 + _ramp_h2:
        return 800.0 + 800.0 * (t - _ramp_h1) / _ramp_h2
    elif t <= _ramp_hold_start:
        return 1600.0 + 50.0 * (t - _ramp_h1 - _ramp_h2) / _ramp_rc
    else:
        return 1650.0 + 10.0 * (t - _ramp_hold_start)

json_file_ramp = vacancy_data_dir / "WC_evolution_ramp_continue_1650_summary.json"

if json_file_ramp.exists():
    try:
        with open(json_file_ramp, "r") as f:
            wc_data = json.load(f)

        time_points = np.array(wc_data["time_points_ns"])
        WC_avg_by_time = np.array(wc_data["WC_avg_by_time"])
        WC_sum_ramp = np.sum(np.abs(WC_avg_by_time), axis=1)
        T_virt = np.array([_vT_ramp(t) for t in time_points])
        is_hold = np.array(wc_data.get("is_hold_phase", [t > _ramp_hold_start for t in time_points]))

        mask_solid = (~is_hold) & (T_virt >= 900.0)
        mask_hold  = is_hold

        T_ramp_solid     = T_virt[mask_solid]
        alpha_ramp_solid = WC_sum_ramp[mask_solid]

        solid_idx = np.where(mask_solid)[0]
        hold_idx  = np.where(mask_hold)[0]
        if len(solid_idx) > 0 and len(hold_idx) > 0:
            conn = np.concatenate([[solid_idx[-1]], hold_idx])
            T_ramp_hold     = T_virt[conn]
            alpha_ramp_hold = WC_sum_ramp[conn]
        elif len(hold_idx) > 0:
            T_ramp_hold     = T_virt[hold_idx]
            alpha_ramp_hold = WC_sum_ramp[hold_idx]

        print(f"  Loaded ramp data: {len(time_points)} points "
              f"({np.sum(mask_solid)} solid, {np.sum(mask_hold)} hold)")
    except Exception as e:
        print(f"  Error loading ramp data: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  JSON file not found: {json_file_ramp}")

################################################################################
# Load single heating ramp simulation data                                    #
################################################################################

print("\nLoading single heating ramp simulation data (s07)...")
alpha_single_solid = None
T_single_solid = None
alpha_single_hold = None
T_single_hold = None

_single_ht, _single_rc = 100.0, 5.0
_single_hold_start = _single_ht + _single_rc  # 105 ns

def _vT_single(t):
    if t <= _single_ht:
        return 600.0 + 1000.0 * t / _single_ht
    elif t <= _single_hold_start:
        return 1600.0 + 50.0 * (t - _single_ht) / _single_rc
    else:
        return 1650.0 + 10.0 * (t - _single_hold_start)

json_file_single = vacancy_data_dir / "WC_evolution_single_ramp_continue_1650_summary.json"

if json_file_single.exists():
    try:
        with open(json_file_single, "r") as f:
            wc_data = json.load(f)

        time_points = np.array(wc_data["time_points_ns"])
        WC_avg_by_time = np.array(wc_data["WC_avg_by_time"])
        WC_sum_single = np.sum(np.abs(WC_avg_by_time), axis=1)
        T_virt = np.array([_vT_single(t) for t in time_points])
        is_hold = np.array(wc_data.get("is_hold_phase", [t > _single_hold_start for t in time_points]))

        mask_solid = (~is_hold) & (T_virt >= 900.0)
        mask_hold  = is_hold

        T_single_solid     = T_virt[mask_solid]
        alpha_single_solid = WC_sum_single[mask_solid]

        solid_idx = np.where(mask_solid)[0]
        hold_idx  = np.where(mask_hold)[0]
        if len(solid_idx) > 0 and len(hold_idx) > 0:
            conn = np.concatenate([[solid_idx[-1]], hold_idx])
            T_single_hold     = T_virt[conn]
            alpha_single_hold = WC_sum_single[conn]
        elif len(hold_idx) > 0:
            T_single_hold     = T_virt[hold_idx]
            alpha_single_hold = WC_sum_single[hold_idx]

        print(f"  Loaded single ramp data: {len(time_points)} points "
              f"({np.sum(mask_solid)} solid, {np.sum(mask_hold)} hold)")
    except Exception as e:
        print(f"  Error loading single ramp data: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  JSON file not found: {json_file_single}")

################################################################################
# Load legacy effective alpha data from old Francisco datasets                 #
################################################################################

print("\nLoading legacy effective alpha data (s07)...")
alpha_old_aged = None
T_old_aged = None
alpha_old_quenched = None
T_old_quenched = None

try:
    xlattice_data = np.load(sim_data_dir / "lattice.npy")
    xlattice = np.mean(xlattice_data, axis=-1)
    _T_list_legacy = np.arange(400, 1601, 100).astype(int)
    WC_1nn_sum_old = np.sum(np.abs(np.load(wc_data_dir / "WC_avg.npy")[0, 1:-3, :]), axis=-1)

    xlattice_sample_old = transform_xlattice_old(_T_list_legacy, TE_list_ref, xlattice)
    (
        A_coeff_old,
        B_coeff_old,
        C_coeff_old,
        shrink_parameter_C_old,
        small_misfit_D_old,
    ) = obtain_fitting_parameters(xlattice_sample_old, WC_1nn_sum_old)

    T_old_aged, lat_old_aged = np.loadtxt("data/Francisco_h1.csv", unpack=True)
    T_old_aged = T_old_aged[:-1]
    lat_old_aged = lat_old_aged[:-1]

    T_old_quenched, lat_old_quenched = np.loadtxt("data/Francisco_h2.csv", unpack=True)
    lat_old_quenched = lat_old_quenched - 0.0018

    alpha_old_aged = (
        C_coeff_old
        + B_coeff_old * T_old_aged
        + A_coeff_old * T_old_aged**2
        + small_misfit_D_old
        - lat_old_aged
    ) / (-shrink_parameter_C_old)
    alpha_old_quenched = (
        C_coeff_old
        + B_coeff_old * T_old_quenched
        + A_coeff_old * T_old_quenched**2
        + small_misfit_D_old
        - lat_old_quenched
    ) / (-shrink_parameter_C_old)
    print(
        "  Loaded legacy alpha data:"
        f" aged={len(T_old_aged)} points, quenched={len(T_old_quenched)} points"
    )
except Exception as e:
    print(f"  Warning: Could not load legacy effective alpha data: {e}")

################################################################################
# Plot alpha profiles                                                          #
################################################################################

print("\nPlotting alpha profiles (s07)...")

fig, ax = plt.subplots(figsize=(2.8, 2.6))

def _plot_sim_series(ax, T_solid, alpha_solid, T_hold, alpha_hold, color, label):
    """Plot a simulation series with solid heating segment and transparent hold segment."""
    if T_solid is not None:
        ax.plot(T_solid, alpha_solid, "-", c=color, linewidth=1.5, label=label, alpha=1.0)
    if T_hold is not None:
        ax.plot(T_hold, alpha_hold, "-", c=color, linewidth=1.5, alpha=0.5)

if WC_spline is not None and T_fit is not None:
    T_eq_plot = T_fit[T_fit >= 700.0]
    if T_eq_plot.size > 0:
        ax.plot(T_eq_plot, WC_spline(T_eq_plot), "-", c="#6B93E4",
                label="Equilibrium", zorder=9, linewidth=1.5)

_plot_sim_series(ax, T_ramp_solid, alpha_ramp_solid, T_ramp_hold, alpha_ramp_hold,
                 "#9ed925", "Synthetic quenched state")
_plot_sim_series(ax, T_single_solid, alpha_single_solid, T_single_hold, alpha_single_hold,
                 "#37b35f", "Synthetic aged state")

ax.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best", fontsize=6.5)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_xlim([700, 1750])
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

plt.tight_layout()
fig_path = output_dir / "s07_alpha_profiles.pdf"
plt.savefig(fig_path, transparent=False)
print(f"  Saved: {fig_path}")
plt.close()

################################################################################
# Plot alpha profiles with legacy effective-alpha data                         #
################################################################################

print("Plotting alpha profiles with legacy effective-alpha data (s07)...")

fig, ax = plt.subplots(figsize=(2.8, 2.6))

if WC_spline is not None and T_fit is not None:
    T_eq_plot = T_fit[T_fit >= 700.0]
    if T_eq_plot.size > 0:
        ax.plot(T_eq_plot, WC_spline(T_eq_plot), "-", c="#6B93E4",
                label="Equilibrium", zorder=9, linewidth=1.5)

_plot_sim_series(ax, T_ramp_solid, alpha_ramp_solid, T_ramp_hold, alpha_ramp_hold,
                 "#9ed925", "Synthetic quenched state")
_plot_sim_series(ax, T_single_solid, alpha_single_solid, T_single_hold, alpha_single_hold,
                 "#37b35f", "Synthetic aged state")

if alpha_old_aged is not None and T_old_aged is not None:
    ax.plot(T_old_aged, alpha_old_aged, "-", c="#37b35f", linewidth=1.5,
            label="Exp aged state", alpha=0.5, zorder=0)

if alpha_old_quenched is not None and T_old_quenched is not None:
    ax.plot(T_old_quenched, alpha_old_quenched, "-", c="#9ed925", linewidth=1.5,
            label="Exp quenched state", alpha=0.5, zorder=0)

ax.set_ylabel(r"CSRO amount $\alpha^{total}$", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="best", fontsize=6.5)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.set_xlim([700, 1750])
ax.yaxis.set_label_coords(-0.10, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

plt.tight_layout()
fig_path = output_dir / "s07_alpha_profiles_with_old.pdf"
plt.savefig(fig_path, transparent=False)
print(f"  Saved: {fig_path}")
plt.close()

################################################################################
# Save summary data for downstream scripts                                     #
################################################################################

summary_data = {}

if T_ramp_solid is not None:
    T_all = np.concatenate([T_ramp_solid, T_ramp_hold[1:]]) if T_ramp_hold is not None else T_ramp_solid
    a_all = np.concatenate([alpha_ramp_solid, alpha_ramp_hold[1:]]) if T_ramp_hold is not None else alpha_ramp_solid
    summary_data["ramp"] = {
        "temperature_K": T_all.tolist(),
        "alpha": a_all.tolist(),
        "is_hold_phase": ([False] * len(T_ramp_solid) + [True] * (len(T_ramp_hold) - 1)) if T_ramp_hold is not None else [False] * len(T_ramp_solid),
    }

if T_single_solid is not None:
    T_all = np.concatenate([T_single_solid, T_single_hold[1:]]) if T_single_hold is not None else T_single_solid
    a_all = np.concatenate([alpha_single_solid, alpha_single_hold[1:]]) if T_single_hold is not None else alpha_single_solid
    summary_data["single_ramp"] = {
        "temperature_K": T_all.tolist(),
        "alpha": a_all.tolist(),
        "is_hold_phase": ([False] * len(T_single_solid) + [True] * (len(T_single_hold) - 1)) if T_single_hold is not None else [False] * len(T_single_solid),
    }

if WC_spline is not None and T_fit is not None:
    T_eq_plot = T_fit[T_fit >= 700.0]
    summary_data["equilibrium"] = {
        "temperature_K": T_eq_plot.tolist(),
        "alpha": WC_spline(T_eq_plot).tolist(),
    }

if alpha_old_aged is not None and T_old_aged is not None:
    summary_data["legacy_aged_state"] = {
        "temperature_K": T_old_aged.tolist(),
        "alpha": alpha_old_aged.tolist(),
    }

if alpha_old_quenched is not None and T_old_quenched is not None:
    summary_data["legacy_quenched_state"] = {
        "temperature_K": T_old_quenched.tolist(),
        "alpha": alpha_old_quenched.tolist(),
    }

summary_path = details_dir / "s07_alpha_profiles_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary_data, f, indent=2)
print(f"  Saved: {summary_path}")

print("\nAnalysis complete (s07_alpha_profiles).")
