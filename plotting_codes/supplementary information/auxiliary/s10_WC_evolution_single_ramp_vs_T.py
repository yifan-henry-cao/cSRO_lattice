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
summary_path = vacancy_data_dir / "WC_evolution_single_ramp_continue_1650_summary.json"

################################################################################
# Load summary                                                                 #
################################################################################

print("Loading WC evolution single-ramp summary (s10)...")

if not summary_path.exists():
    raise FileNotFoundError(
        f"Required summary file not found: {summary_path}. "
        "Please generate it in the Monte Carlo workflow and copy via 00_gather_data.py."
    )

with open(summary_path, "r") as f:
    data = json.load(f)

time_points = np.array(data["time_points_ns"], dtype=float)
WC_avg_by_time = np.array(data["WC_avg_by_time"], dtype=float)
WC_ste_by_time = np.array(data["WC_ste_by_time"], dtype=float)
pair_labels = data["pair_labels"]
heating_time = float(data.get("heating_time_ns", 100.0))

n_pairs = WC_avg_by_time.shape[1]

# Equilibrium CSRO values: reuse ramp-equilibrium data for consistency
T_eq = None
WC_eq = None

try:
    ramp_summary_path = vacancy_data_dir / "WC_evolution_ramp_continue_1650_summary.json"
    with open(ramp_summary_path, "r") as f_ramp:
        ramp_data = json.load(f_ramp)

    WC_equilibrium_avg_by_T = ramp_data.get("WC_equilibrium_avg_by_T", None)
    if WC_equilibrium_avg_by_T is not None:
        T_eq_list = sorted(float(T) for T in WC_equilibrium_avg_by_T.keys())
        T_eq = np.array(T_eq_list, dtype=float)
        eq_values = [WC_equilibrium_avg_by_T[str(int(T))] for T in T_eq_list]
        WC_eq = np.array(eq_values, dtype=float)
except Exception:
    T_eq = None
    WC_eq = None


_s_ht, _s_rc = heating_time, 5.0
_s_hold_start = _s_ht + _s_rc  # 105 ns

def time_to_temperature(t_ns: float) -> float:
    """Virtual temperature: heating + continuation ramp, hold extrapolated at 10 K/ns."""
    if t_ns <= _s_ht:
        return 600.0 + 1000.0 * t_ns / _s_ht
    elif t_ns <= _s_hold_start:
        return 1600.0 + 50.0 * (t_ns - _s_ht) / _s_rc
    else:
        return 1650.0 + 10.0 * (t_ns - _s_hold_start)


T = np.array([time_to_temperature(t) for t in time_points])
is_hold = np.array(data.get("is_hold_phase", [t > _s_hold_start for t in time_points]), dtype=bool)

################################################################################
# Plot WC evolution vs temperature                                             #
################################################################################

print("\nPlotting WC evolution vs temperature (s10)...")

fig, ax = plt.subplots(figsize=(2.8, 2.6))

colors = ["#8c564b", "#90C47F", "#FE7C46", "#21B0FE", "#9069C5", "#FE218B"]
while len(colors) < n_pairs:
    colors.extend(colors)

solid_idx = np.where(~is_hold)[0]
hold_idx  = np.where(is_hold)[0]
conn = np.concatenate([[solid_idx[-1]], hold_idx]) if (len(solid_idx) > 0 and len(hold_idx) > 0) else hold_idx

for pair_idx in range(n_pairs):
    mean = WC_avg_by_time[:, pair_idx]
    ste  = WC_ste_by_time[:, pair_idx]
    c    = colors[pair_idx]

    ax.plot(T[~is_hold], mean[~is_hold], "-", color=c, linewidth=1.2,
            label=pair_labels[pair_idx])
    ax.fill_between(T[~is_hold], mean[~is_hold] - ste[~is_hold],
                    mean[~is_hold] + ste[~is_hold], facecolor=c, alpha=0.2)

    if len(hold_idx) > 0:
        ax.plot(T[conn], mean[conn], "-", color=c, linewidth=1.2, alpha=0.5)
        ax.fill_between(T[conn], mean[conn] - ste[conn], mean[conn] + ste[conn],
                        facecolor=c, alpha=0.1)

    if T_eq is not None and WC_eq is not None:
        ax.plot(T_eq[2:], WC_eq[2:, pair_idx], "--", color=c, linewidth=0.9, alpha=0.8)

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, linestyle="-", zorder=0)

ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"CSRO parameter $\alpha_{ij}$", fontsize=8)
ax.set_xlim([600, 1700])
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.legend(loc="best", fontsize=6, ncol=2, frameon=True)
ax.xaxis.set_label_coords(0.5, -0.08)
ax.yaxis.set_label_coords(-0.12, 0.5)

plt.tight_layout()
fig_path = output_dir / "s10_WC_evolution_single_ramp_vs_T.pdf"
plt.savefig(fig_path, transparent=False)
print(f"  Saved: {fig_path}")
plt.close()

################################################################################
# Save summary for downstream scripts                                          #
################################################################################

summary_out = {
    "temperature_K": T.tolist(),
    "WC_avg_by_time": WC_avg_by_time.tolist(),
    "WC_ste_by_time": WC_ste_by_time.tolist(),
    "pair_labels": pair_labels,
}

summary_details_path = details_dir / "s10_WC_evolution_single_ramp_vs_T_summary.json"
with open(summary_details_path, "w") as f:
    json.dump(summary_out, f, indent=2)
print(f"  Saved: {summary_details_path}")

print("\nAnalysis complete (s10_WC_evolution_single_ramp_vs_T).")
