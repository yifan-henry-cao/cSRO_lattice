# cSRO_lattice
cSRO_lattice is a repository containing the trained Machine-Learning Potentials, atomistic simulation data and relevant plotting codes published with our work ``On the nature of chemical short-range order evolution``

The atom type mapping for the potential, training dataset, and simulation data are: {0: 'Cr', 1: 'Co', 2: 'Ni'}

The final snapshots from Monte Carlo simulations that are used as input structures for Molecular Dynamic simulations are contained in ./simulations/MC_structures/, while a fraction of the final structures from MD simulations (the ones used to evaluate bond lengths) are contained in ./simulations/ordered_thermal_expansion/. Additional atomistic simulation data can be added to this repository upon reasonable request.

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
@misc{caocapturing2024,
      title={Capturing short-range order in high-entropy alloys with machine learning potentials}, 
      author={Yifan Cao and Killian Sheriff and Rodrigo Freitas},
      year={2024},
      eprint={2401.06622},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2401.06622}, 
}
```