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
ramp_summary_path = vacancy_data_dir / "WC_evolution_ramp_continue_1650_summary.json"
single_ramp_summary_path = vacancy_data_dir / "WC_evolution_single_ramp_continue_1650_summary.json"

################################################################################
# Helpers                                                                      #
################################################################################


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required summary file not found: {path}. "
            "Please generate it in the Monte Carlo workflow and copy via 00_gather_data.py."
        )
    with open(path, "r") as f:
        return json.load(f)


def find_nini_pair_index(pair_labels) -> int:
    normalized = [label.replace(" ", "").replace("_", "").replace("/", "").lower() for label in pair_labels]
    candidates = ("nini", "ni-ni", "ninii")
    for idx, label in enumerate(normalized):
        if any(candidate in label for candidate in candidates):
            return idx
    raise ValueError(f"Could not find Ni-Ni pair in pair_labels: {pair_labels}")


def ramp_time_to_temperature(t_ns: float) -> float:
    h1, h2, rc = 4.0, 80.0, 5.0
    hold_start = h1 + h2 + rc  # 89 ns
    if t_ns <= h1:
        return 400.0 + 400.0 * t_ns / h1
    elif t_ns <= h1 + h2:
        return 800.0 + 800.0 * (t_ns - h1) / h2
    elif t_ns <= hold_start:
        return 1600.0 + 50.0 * (t_ns - h1 - h2) / rc
    else:
        return 1650.0 + 10.0 * (t_ns - hold_start)


def single_ramp_time_to_temperature(t_ns: float, heating_time: float) -> float:
    rc = 5.0
    hold_start = heating_time + rc  # 105 ns
    if t_ns <= heating_time:
        return 600.0 + 1000.0 * t_ns / heating_time
    elif t_ns <= hold_start:
        return 1600.0 + 50.0 * (t_ns - heating_time) / rc
    else:
        return 1650.0 + 10.0 * (t_ns - hold_start)


################################################################################
# Load summaries                                                               #
################################################################################

print("Loading Ni-Ni WC evolution summaries (s11)...")

ramp_data = load_json(ramp_summary_path)
single_ramp_data = load_json(single_ramp_summary_path)

pair_labels = ramp_data["pair_labels"]
pair_idx = find_nini_pair_index(pair_labels)
pair_label = pair_labels[pair_idx]

time_points_ramp = np.array(ramp_data["time_points_ns"], dtype=float)
WC_avg_ramp = np.array(ramp_data["WC_avg_by_time"], dtype=float)
WC_ste_ramp = np.array(ramp_data["WC_ste_by_time"], dtype=float)

time_points_single = np.array(single_ramp_data["time_points_ns"], dtype=float)
WC_avg_single = np.array(single_ramp_data["WC_avg_by_time"], dtype=float)
WC_ste_single = np.array(single_ramp_data["WC_ste_by_time"], dtype=float)
heating_time_single = float(single_ramp_data.get("heating_time_ns", 100.0))

if single_ramp_data["pair_labels"] != pair_labels:
    raise ValueError("Pair labels differ between ramp and single-ramp summaries.")

T_ramp_all = np.array([ramp_time_to_temperature(t) for t in time_points_ramp])
T_single_all = np.array([single_ramp_time_to_temperature(t, heating_time_single) for t in time_points_single])

is_hold_ramp   = np.array(ramp_data.get("is_hold_phase",   [False]*len(time_points_ramp)),   dtype=bool)
is_hold_single = np.array(single_ramp_data.get("is_hold_phase", [False]*len(time_points_single)), dtype=bool)

# Skip first 40 frames of ramp (low-temperature startup)
skip = 40
T_ramp_all   = T_ramp_all[skip:]
is_hold_ramp = is_hold_ramp[skip:]
WC_avg_ramp  = WC_avg_ramp[skip:]
WC_ste_ramp  = WC_ste_ramp[skip:]

mean_ramp   = WC_avg_ramp[:, pair_idx]
ste_ramp    = WC_ste_ramp[:, pair_idx]
mean_single = WC_avg_single[:, pair_idx]
ste_single  = WC_ste_single[:, pair_idx]

T_ramp   = T_ramp_all
T_single = T_single_all

T_eq = None
WC_eq = None
WC_equilibrium_avg_by_T = ramp_data.get("WC_equilibrium_avg_by_T", None)
if WC_equilibrium_avg_by_T is not None:
    T_eq_list = sorted(float(T) for T in WC_equilibrium_avg_by_T.keys())
    T_eq = np.array(T_eq_list, dtype=float)
    WC_eq = np.array(
        [WC_equilibrium_avg_by_T[str(int(T))][pair_idx] for T in T_eq_list],
        dtype=float,
    )

################################################################################
# Plot Ni-Ni evolution vs temperature                                          #
################################################################################

print("\nPlotting Ni-Ni WC evolution vs temperature (s11)...")

fig, ax = plt.subplots(figsize=(3.0, 2.6))

def _plot_series(ax, T_all, mean, ste, is_hold, color, label):
    solid_idx = np.where(~is_hold)[0]
    hold_idx  = np.where(is_hold)[0]
    ax.plot(T_all[~is_hold], mean[~is_hold], "-", color=color, linewidth=1.5, label=label)
    ax.fill_between(T_all[~is_hold], mean[~is_hold] - ste[~is_hold],
                    mean[~is_hold] + ste[~is_hold], facecolor=color, alpha=0.2)
    if len(hold_idx) > 0:
        conn = np.concatenate([[solid_idx[-1]], hold_idx]) if len(solid_idx) > 0 else hold_idx
        ax.plot(T_all[conn], mean[conn], "-", color=color, linewidth=1.5, alpha=0.5)
        ax.fill_between(T_all[conn], mean[conn] - ste[conn], mean[conn] + ste[conn],
                        facecolor=color, alpha=0.1)

_plot_series(ax, T_ramp,   mean_ramp,   ste_ramp,   is_hold_ramp,   "#9ed925", "Synthetic quenched state")
_plot_series(ax, T_single, mean_single, ste_single, is_hold_single, "#37b35f", "Synthetic aged state")

if T_eq is not None and WC_eq is not None:
    ax.plot(T_eq[2:], WC_eq[2:], "--", color="#6B93E4", linewidth=1.2, alpha=0.9, label="Equilibrium")

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, linestyle="-", zorder=0)
ax.axhline(mean_ramp[0],   color="#9ed925", linewidth=0.8, alpha=0.5, linestyle="--", zorder=0)
ax.axhline(mean_single[0], color="#37b35f", linewidth=0.8, alpha=0.5, linestyle="--", zorder=0)

ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Ni-Ni CSRO parameter $\alpha_{Ni-Ni}$", fontsize=8)
ax.set_xlim([600, 1700])
ax.set_ylim([-0.095, 0.005])
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.legend(loc="best", fontsize=6.5, frameon=True)
ax.xaxis.set_label_coords(0.5, -0.08)
ax.yaxis.set_label_coords(-0.16, 0.5)

plt.tight_layout()
fig_path = output_dir / "s11_WC_NiNi_evolution.pdf"
plt.savefig(fig_path, transparent=False)
print(f"  Saved: {fig_path}")
plt.close()

################################################################################
# Save summary for downstream scripts                                          #
################################################################################

summary_out = {
    "pair_label": pair_label,
    "synthetic_quenched": {
        "temperature_K": T_ramp.tolist(),
        "WC_avg": mean_ramp.tolist(),
        "WC_ste": ste_ramp.tolist(),
    },
    "synthetic_aged": {
        "temperature_K": T_single.tolist(),
        "WC_avg": mean_single.tolist(),
        "WC_ste": ste_single.tolist(),
    },
}

if T_eq is not None and WC_eq is not None:
    summary_out["equilibrium"] = {
        "temperature_K": T_eq.tolist(),
        "WC_avg": WC_eq.tolist(),
    }

summary_details_path = details_dir / "s11_WC_NiNi_evolution_summary.json"
with open(summary_details_path, "w") as f:
    json.dump(summary_out, f, indent=2)
print(f"  Saved: {summary_details_path}")

print("\nAnalysis complete (s11_WC_NiNi_evolution).")
