#!/bin/bash
set -e
cd "$(dirname "$0")"

for script in 01_lattice_summary.py 02_thermal_expansion.py 03_lattice_constants.py \
              04_effective_alpha.py 05_CSRO_contribution.py 06_CSRO_lattice.py \
              07_bond_length_analysis.py 08_WC_vs_T.py; do
    echo "Running $script..."
    python "$script"
done

echo "Done."
