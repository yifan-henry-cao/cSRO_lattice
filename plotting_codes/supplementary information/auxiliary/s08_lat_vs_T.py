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
summary_input = vacancy_data_dir / "s02_lat_vs_T_summary.json"

################################################################################
# Load precomputed lattice vs T summary                                        #
################################################################################

print("Loading precomputed lattice vs T summary (s08)...")

if not summary_input.exists():
    raise FileNotFoundError(
        f"Required summary file not found: {summary_input}. "
        "Please generate it in the Monte Carlo post-processing workflow "
        "and copy it via 00_gather_data.py."
    )

with open(summary_input, "r") as f:
    data = json.load(f)

window_size = data.get("window_size_snapshots", None)

ramp = data.get("ramp", {})
single_ramp = data.get("single_ramp", {})
aged_fits = data.get("aged_state_fits", {})

ramp_T = np.array(ramp.get("temperature_K", []), dtype=float)
ramp_a = np.array(ramp.get("lattice_parameter_Ang", []), dtype=float)

single_T = np.array(single_ramp.get("temperature_K", []), dtype=float)
single_a = np.array(single_ramp.get("lattice_parameter_Ang", []), dtype=float)

fit1_info = aged_fits.get("fit1_900_1100K", None)
fit2_info = aged_fits.get("fit2_1500_1600K", None)
Tkr = aged_fits.get("Tkr_K", None)

if fit1_info is not None and fit2_info is not None and Tkr is not None:
    m1 = float(fit1_info["slope"])
    b1 = float(fit1_info["intercept"])
    m2 = float(fit2_info["slope"])
    b2 = float(fit2_info["intercept"])
else:
    m1 = b1 = m2 = b2 = None
    print("  Warning: aged_state_fits not fully specified in summary; fits will be skipped.")

################################################################################
# Plot lattice vs temperature above 900 K                                      #
################################################################################

print("\nPlotting lattice parameter vs temperature (s08)...")

fig, ax = plt.subplots(figsize=(2.8, 2.6))

# Ramp (Synthetic quenched state), T >= 900 K
mask_ramp = ramp_T >= 900.0
if np.any(mask_ramp):
    ax.plot(
        ramp_T[mask_ramp],
        ramp_a[mask_ramp],
        "-",
        c="#9ed925",
        linewidth=0.7,
        label="Synthetic quenched state",
        alpha=1.0,
    )

# Single ramp (Synthetic aged state), T >= 900 K
mask_single = single_T >= 900.0
if np.any(mask_single):
    ax.plot(
        single_T[mask_single],
        single_a[mask_single],
        "-",
        c="#37b35f",
        linewidth=0.7,
        label="Synthetic aged state",
        alpha=1.0,
    )

ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Lattice parameter (Å)", fontsize=8)
ax.set_xlim([900, 1600])
ax.text(
    0.44,
    0.1,
    r"$<\Delta a> = 5.6\times 10^{-4}\ \mathrm{\AA}$",
    transform=ax.transAxes,
    fontsize=7,
    va="top",
    ha="left",
)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.legend(loc="best", fontsize=7)
ax.yaxis.set_label_coords(-0.14, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

plt.tight_layout()
fig_path = output_dir / "s08_lat_vs_T.pdf"
plt.savefig(fig_path, transparent=False)
print(f"  Saved: {fig_path}")
plt.close()

################################################################################
# Save summary data for downstream scripts                                     #
################################################################################

summary_out = {
    "window_size_snapshots": window_size,
    "ramp": {
        "temperature_K": ramp_T[mask_ramp].tolist(),
        "lattice_parameter_Ang": ramp_a[mask_ramp].tolist(),
    },
    "single_ramp": {
        "temperature_K": single_T[mask_single].tolist(),
        "lattice_parameter_Ang": single_a[mask_single].tolist(),
    },
}

if m1 is not None and m2 is not None and Tkr is not None:
    summary_out["aged_state_fits"] = {
        "fit1_900_1100K": {"slope": m1, "intercept": b1},
        "fit2_1500_1600K": {"slope": m2, "intercept": b2},
        "Tkr_K": Tkr,
    }

summary_path = details_dir / "s08_lat_vs_T_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary_out, f, indent=2)
print(f"  Saved: {summary_path}")

print("\nAnalysis complete (s08_lat_vs_T).")
