import numpy as np
import os

os.system("mkdir -p ./data/WC_params/")

# Move experimental raw data to current folder
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/ordered_thermal_expansion/post_processing/data/* ./data/")

# Move computational WC data to current folder
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/post_processing/data/WC_avg.npy ./data/WC_params/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/post_processing/data/WC_std.npy ./data/WC_params/")

# Move bond length data to current folder
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/ordered_thermal_expansion/post_processing/lattice_data/bond_length_TE.npy ./data/sim_data/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/ordered_thermal_expansion/post_processing/lattice_data/bond_length_TSRO.npy ./data/sim_data/")

# Process simulation data
def ReadData(T0, T, ID):
    # Load simulation data.
    pe, xlat = np.loadtxt(
        f"/home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/ordered_thermal_expansion/data/thermo/thermo_T0_{T0}K_T_{T}K_{ID}.dat",
        unpack=True,
        usecols=(3, 4),
    )
    half_point = len(xlat) // 2

    # Compute thermodynamic averages (skips first 1ps while the system equilibrates)
    xlat_avg = xlat[half_point:].mean()
    pe_avg = pe[half_point:].mean()

    return pe_avg, xlat_avg

T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)
ID_list = np.array(range(1, 10))  # Independent runs
xlattice_data = np.zeros((len(T_list), len(TE_list), len(ID_list)))
potential_data = np.zeros((len(T_list), len(TE_list), len(ID_list)))

for idx, T0 in enumerate(T_list):
    print(T0)
    for TE_idx, T in enumerate(TE_list):
        for ID_idx, ID in enumerate(ID_list):
            potential_data[idx, TE_idx, ID_idx], xlattice_data[idx, TE_idx, ID_idx] = ReadData(T0, T, ID)

potential = np.mean(potential_data, axis=-1)  # (13, 33, 10) -> (13, 33)
xlattice = np.mean(xlattice_data, axis=-1)

os.system("mkdir -p data/sim_data")
np.save("data/sim_data/potential.npy", potential_data)  # Shape (13, 33)
np.save("data/sim_data/lattice.npy", xlattice_data)

def compute_CTE(T_list, xlattice):
    """
    Compute Coefficient of Thermal Expansion using 5-point central difference method

    Parameters:
    -----------
    T_list : numpy array
        Temperature points
    xlattice : numpy array
        Lattice parameters, shape (n_configs, n_temps)

    Returns:
    --------
    T_central : numpy array
        Temperature points where CTE is evaluated (excludes 2 points from each end)
    sim_CTE : numpy array
        CTE values computed using 5-point stencil
    """
    # Need at least 5 points for 5-point stencil
    if len(T_list) < 5:
        raise ValueError("Need at least 5 temperature points for 5-point central difference")

    # Five-point stencil coefficients for first derivative
    # f'(x) ≈ (f(x-2h) - 8f(x-h) + 0f(x) + 8f(x+h) - f(x+2h))/(12h)
    coeff = np.array([1, -8, 0, 8, -1]) / 12.0

    # Initialize arrays for CTE calculation
    n_configs = xlattice.shape[0]
    n_central_points = len(T_list) - 4  # Exclude 2 points from each end
    sim_CTE = np.zeros((n_configs, n_central_points))
    T_central = T_list[2:-2]  # Central points where CTE is evaluated

    for i in range(n_central_points):
        # Get 5 consecutive points centered at i+2
        L_window = xlattice[:, i:i+5]  # Shape (13, 5)
        T_window = T_list[i:i+5]

        # Calculate dT (spacing between points)
        dT = T_window[1] - T_window[0]  # Assuming uniform temperature spacing

        # Calculate derivative using 5-point stencil
        dL = np.sum(coeff * L_window, axis=1)  # (5,) * (13, 5)

        # Calculate CTE = (1/L)(dL/dT)
        L_central = xlattice[:, i+2]  # Use central point for L
        sim_CTE[:, i] = (dL / dT) / L_central * 1e6  # Convert to ppm/K

    return T_central, sim_CTE

T_central, sim_CTE = compute_CTE(TE_list, xlattice)
np.save("data/sim_data/thermal_expansion.npy", sim_CTE)
