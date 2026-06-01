# cSRO_lattice
cSRO_lattice is a repository containing the trained Machine-Learning Potentials, atomistic simulation data and relevant plotting codes published with our work ``On the nature of chemical short-range order evolution``

The atom type mapping for the potential, training dataset, and simulation data are: {0: 'Cr', 1: 'Co', 2: 'Ni'}

The final snapshots from Monte Carlo simulations that are used as input structures for Molecular Dynamic simulations are contained in ./simulations/MC_structures/, while a fraction of the final structures from MD simulations (the ones used to evaluate bond lengths) are contained in ./simulations/ordered_thermal_expansion/. Additional atomistic simulation data can be added to this repository upon reasonable request.

All source codes for reproducing Fig.3 and Fig.4 of the manuscript and Supplementary Figure.8-15 of the supplementary information can be found in ./plotting_codes/ using the run_all.sh code within auxiliary folders.

## References & Citing
If you use this repository in your work, please cite:

```
@misc{cSRO_lattice,
      title={On the nature of chemical short-range order evolution}, 
      author={G.C. Stumpf, Y. Cao, V.P. Bacurau, D. Miracle, W. Wolf, E.D. Zanotto, R. Freitas, F.G. Coury},
      year={2025},
}
```

and 

```
@article{cao_capturing_2025,
  title = {Capturing Short-Range Order in High-Entropy Alloys with Machine Learning Potentials},
  author = {Cao, Yifan and Sheriff, Killian and Freitas, Rodrigo},
  year = 2025,
  month = aug,
  journal = {npj Computational Materials},
  volume = {11},
  number = {1},
  pages = {268},
  issn = {2057-3960},
  doi = {10.1038/s41524-025-01722-2},
  urldate = {2025-08-21},
  copyright = {All rights reserved},
  langid = {english}
}
```