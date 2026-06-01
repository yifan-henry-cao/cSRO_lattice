import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("paper")

################################################################################
# Paths and configuration                                                      #
################################################################################

output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
details_dir = output_dir / "details"
details_dir.mkdir(parents=True, exist_ok=True)

vacancy_data_dir = Path("data/vacancy_jump_heating")
summary_path = vacancy_data_dir / "vacancy_jumps_single_ramp_continue_1650_summary.json"
alpha_profiles_summary_path = details_dir / "s07_alpha_profiles_summary.json"

_single_hold_start = 105.0  # ns

################################################################################
# Load summary                                                                 #
################################################################################

print("Loading vacancy jump single-ramp summary (s09)...")

if not summary_path.exists():
    raise FileNotFoundError(
        f"Required summary file not found: {summary_path}. "
        "Please generate it in the Monte Carlo workflow and copy via 00_gather_data.py."
    )

with open(summary_path, "r") as f:
    data = json.load(f)

time_pts = np.array(data["time_points_ns"], dtype=float)
is_hold = np.array(data.get("is_hold_phase", [False] * len(time_pts)), dtype=bool)
jumps = np.array(data["average_jumps_per_ps_corrected"], dtype=float)
ste_jumps = np.array(data["ste_jumps_per_ps_corrected"], dtype=float)

# Build virtual temperature: hold phase extrapolated at 10 K/ns past 1650K
_rc = 5.0
def _vT(t):
    if t <= 100.0:
        return 600.0 + 1000.0 * t / 100.0
    elif t <= _single_hold_start:
        return 1600.0 + 50.0 * (t - 100.0) / _rc
    else:
        return 1650.0 + 10.0 * (t - _single_hold_start)

T_virtual = np.array([_vT(t) for t in time_pts])
T_real = T_virtual  # use virtual T as x-axis for plotting

if not alpha_profiles_summary_path.exists():
    raise FileNotFoundError(
        f"Required alpha-profile summary file not found: {alpha_profiles_summary_path}. "
        "Please run s07_alpha_profiles.py first."
    )

with open(alpha_profiles_summary_path, "r") as f:
    alpha_profiles_data = json.load(f)

single_ramp_data = alpha_profiles_data.get("single_ramp")
if single_ramp_data is None:
    raise KeyError(
        "Synthetic aged-state data ('single_ramp') not found in "
        f"{alpha_profiles_summary_path}."
    )

T_alpha = np.array(single_ramp_data["temperature_K"], dtype=float)
alpha_aged = np.array(single_ramp_data["alpha"], dtype=float)
dalpha_dT = np.gradient(alpha_aged, T_alpha)
smooth_window = 50
if dalpha_dT.size >= smooth_window:
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    dalpha_dT_smoothed = np.convolve(dalpha_dT, kernel, mode="valid")
    T_alpha = np.convolve(T_alpha, kernel, mode="valid")
else:
    dalpha_dT_smoothed = dalpha_dT.copy()

################################################################################
# Plot vacancy jumps vs temperature                                            #
################################################################################

print("\nPlotting vacancy jump frequency vs temperature (s09)...")

fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax2 = ax.twinx()
left_color = "#FE218B"
right_color = "#21B0FE"

mask_solid = ~is_hold
mask_hold  = is_hold

ax.plot(T_real[mask_solid], jumps[mask_solid], "-", linewidth=1.5, color=left_color, alpha=0.9)
ax.fill_between(T_real[mask_solid], jumps[mask_solid] - ste_jumps[mask_solid],
                jumps[mask_solid] + ste_jumps[mask_solid], facecolor=left_color, alpha=0.2)

if np.any(mask_hold):
    solid_idx = np.where(mask_solid)[0]
    hold_idx  = np.where(mask_hold)[0]
    conn = np.concatenate([[solid_idx[-1]], hold_idx]) if len(solid_idx) > 0 else hold_idx
    ax.plot(T_real[conn], jumps[conn], "-", linewidth=1.5, color=left_color, alpha=0.45)
    ax.fill_between(T_real[conn], jumps[conn] - ste_jumps[conn],
                    jumps[conn] + ste_jumps[conn], facecolor=left_color, alpha=0.1)

ax2.plot(T_alpha, dalpha_dT_smoothed * 1E4, "-", linewidth=1.3, color=right_color, alpha=0.95)

ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Vacancy jump frequency (ps$^{-1}$)", fontsize=8)
ax2.set_ylabel(r"$d\alpha^{total}/dT$ (10$^{-4}$ K$^{-1}$)", fontsize=8)
ax.tick_params(axis="y", colors=left_color)
ax2.tick_params(axis="y", colors=right_color)
ax.yaxis.label.set_color(left_color)
ax2.yaxis.label.set_color(right_color)
ax.spines["left"].set_color(left_color)
ax2.spines["left"].set_color(left_color)
ax2.spines["right"].set_color(right_color)
ax.set_ylim([0, 2.0])
ax2.set_ylim([-17, 0.5])
ax.set_xlim([850, 1750])
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(True)
ax.xaxis.set_label_coords(0.5, -0.08)
ax.yaxis.set_label_coords(-0.15, 0.5)
ax2.yaxis.set_label_coords(1.14, 0.5)

plt.tight_layout()
fig_path = output_dir / "s09_vacancy_jumps_vs_T_single_ramp.pdf"
plt.savefig(fig_path, transparent=False)
print(f"  Saved: {fig_path}")
plt.close()

################################################################################
# Save summary for downstream scripts                                          #
################################################################################

summary_out = {
    "temperature_K": T_virtual.tolist(),
    "is_hold_phase": is_hold.tolist(),
    "average_jumps_per_ps_corrected": jumps.tolist(),
    "ste_jumps_per_ps_corrected": ste_jumps.tolist(),
    "alpha_derivative_temperature_K": T_alpha.tolist(),
    "synthetic_aged_state_alpha_total": alpha_aged.tolist(),
    "synthetic_aged_state_dalpha_dT": dalpha_dT.tolist(),
    "synthetic_aged_state_dalpha_dT_smoothed_window_100": dalpha_dT_smoothed.tolist(),
}

summary_details_path = details_dir / "s09_vacancy_jumps_vs_T_single_ramp_summary.json"
with open(summary_details_path, "w") as f:
    json.dump(summary_out, f, indent=2)
print(f"  Saved: {summary_details_path}")

print("\nAnalysis complete (s09_vacancy_jumps_vs_T_single_ramp).")
