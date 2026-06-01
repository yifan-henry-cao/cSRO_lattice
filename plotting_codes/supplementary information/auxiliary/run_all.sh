#!/bin/bash
set -e
cd "$(dirname "$0")"

for script in s01_bond_length_analysis.py \
              s02_lattice_vs_alpha.py \
              s03_WC_vs_T.py \
              s04_lattice_constants.py \
              s05_thermal_expansion.py \
              s06_lattice_vs_T_selected_CSRO.py \
              s07_alpha_profiles.py \
              s08_lat_vs_T.py \
              s09_vacancy_jumps_vs_T_single_ramp.py \
              s10_WC_evolution_single_ramp_vs_T.py \
              s11_WC_NiNi_evolution.py \
              s12_Tkr_concept_Supplementary_Figures_8_9.py; do
    echo "Running $script..."
    python "$script"
done

echo "Done."
