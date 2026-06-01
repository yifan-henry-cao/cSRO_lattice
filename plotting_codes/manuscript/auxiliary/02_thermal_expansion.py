import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import os

plt.style.use("paper")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullLocator

from scipy.optimize import curve_fit

os.system("mkdir -p figures/")


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
    n_central_points = len(T_list) - 4  # Exclude 2 points from each end
    sim_CTE = np.zeros((n_central_points))
    T_central = T_list[2:-2]  # Central points where CTE is evaluated

    for i in range(n_central_points):
        # Get 5 consecutive points centered at i+2
        L_window = xlattice[i:i+5]  # Shape (5)
        T_window = T_list[i:i+5]

        # Calculate dT (spacing between points)
        dT = T_window[1] - T_window[0]  # Assuming uniform temperature spacing

        # Calculate derivative using 5-point stencil
        dL = np.sum(coeff * L_window)  # (5,) * (5)

        # Calculate CTE = (1/L)(dL/dT)
        L_central = xlattice[i+2]  # Use central point for L
        sim_CTE[i] = (dL / dT) / L_central * 1e6  # Convert to ppm/K

    return T_central, sim_CTE


def compute_derivative_5point(x, y):
    """
    Compute derivative using 5-point central difference method for non-uniform grid

    Parameters:
    -----------
    x : numpy array
        Independent variable points (can be non-uniform)
    y : numpy array
        Dependent variable values

    Returns:
    --------
    x_central : numpy array
        Points where derivative is evaluated (excludes 2 points from each end)
    dy_dx : numpy array
        Derivative values computed using 5-point stencil
    """
    if len(x) < 5:
        raise ValueError("Need at least 5 points for 5-point central difference")

    n_central_points = len(x) - 4
    dy_dx = np.zeros(n_central_points)
    x_central = x[2:-2]

    for i in range(n_central_points):
        # Get 5 consecutive points
        x_window = x[i:i+5]
        y_window = y[i:i+5]

        # Calculate distances from central point (x[i+2])
        h = x_window - x_window[2]  # h = [h2, h1, 0, h3, h4]

        # Construct matrix A from Taylor expansion
        # Each row represents coefficients for: f, f', f'', f''', f⁽⁴⁾
        A = np.array([
            [1, 1, 1, 1, 1],                                                # f term
            [h[0], h[1], 0, h[3], h[4]],                                    # f' term
            [h[0]**2, h[1]**2, 0, h[3]**2, h[4]**2],                       # f'' term
            [h[0]**3, h[1]**3, 0, h[3]**3, h[4]**3],                       # f''' term
            [h[0]**4, h[1]**4, 0, h[3]**4, h[4]**4]                        # f⁽⁴⁾ term
        ])

        # Right-hand side vector: we want coefficients that give f'
        b = np.array([0, 1, 0, 0, 0])

        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            coeffs = np.linalg.pinv(A) @ b

        # Compute derivative using the coefficients
        dy_dx[i] = np.dot(coeffs, y_window)

    return x_central, dy_dx


def compute_CTE_nonuniform(T_list, xlattice):
    # Need at least 5 points for 5-point stencil
    if len(T_list) < 5:
        raise ValueError("Need at least 5 temperature points for 5-point central difference")

    T_central, dL_dT = compute_derivative_5point(T_list, xlattice)

    L_central = xlattice[2:-2]
    exp_CTE = (dL_dT) / L_central * 1e6  # Convert to ppm/K

    return T_central, exp_CTE


def convolve_1D(arr, window_size=10):
    kernel = np.ones(window_size) / window_size
    averaged_array = np.convolve(arr, kernel, mode="valid")
    return averaged_array


def generate_pure_polynomial_function(order=3):
    """
    Generates a custom function with pure polynomial terms up to specified order
    for x and y variables separately (no cross terms).

    Parameters:
    -----------
    order : int
        Maximum order of polynomial terms

    Returns:
    --------
    custom_function : function
        Generated function that takes X ([x,y]) and coefficients as arguments
    param_names : list
        List of parameter names for the coefficients
    """

    def get_term_string(power, var, param):
        """Helper function to generate term string"""
        if power == 0:
            return param
        return f"{param}*{var}" + (f"**{power}" if power > 1 else "")

    # Generate parameter names and terms
    param_names = []
    term_strings = []
    coef_idx = 0

    # Constant term
    param_name = f"p{coef_idx}"
    param_names.append(param_name)
    term_strings.append(param_name)
    coef_idx += 1

    # Generate pure x terms
    for power in range(1, order + 1):
        param_name = f"p{coef_idx}"
        param_names.append(param_name)
        term_strings.append(get_term_string(power, 'x', param_name))
        coef_idx += 1

    # Generate pure y terms
    for power in range(1, order + 1):
        param_name = f"p{coef_idx}"
        param_names.append(param_name)
        term_strings.append(get_term_string(power, 'y', param_name))
        coef_idx += 1

    # Create function definition string
    func_def = f"""def custom_function(X, {', '.join(param_names)}):
    x, y = X
    return {' + '.join(term_strings)}"""

    docstring = f"""
    Pure polynomial function of order {order} in x and y (no cross terms).
    Function form: z = {' + '.join(term_strings)}
    """

    # Create function namespace
    namespace = {}
    exec(func_def, namespace)
    custom_function = namespace['custom_function']
    custom_function.__doc__ = docstring

    return custom_function, param_names


def fit_custom_function(custom_function, T_list, TE_list, xlattice):
    """Fit a custom function to 2D data"""
    # Create meshgrid and flatten
    T_mesh, TE_mesh = np.meshgrid(T_list, TE_list, indexing='ij')
    X = [T_mesh.flatten(), TE_mesh.flatten()]
    z = xlattice.flatten()

    # Fit using curve_fit
    popt, pcov = curve_fit(custom_function, X, z)

    return lambda x, y: custom_function([x, y], *popt), popt


################################################################################
# Load and process data.                                                       #
################################################################################
T_list = np.arange(400, 1601, 100).astype(int)
TE_list = np.arange(400, 1201, 25).astype(int)
ID_list = np.array(range(1, 10))  # Independent runs

xlattice_data = np.load("data/sim_data/lattice.npy")
xlattice = mean(xlattice_data, axis=-1)  # (13, 33, 10) -> (13, 33)

from scipy.interpolate import RectBivariateSpline

# Generate mesh grid
x, y = np.meshgrid(T_list, TE_list)

# Perform 2D polynomial
custom_func, params = generate_pure_polynomial_function(order=5)
fitted_function, popt = fit_custom_function(custom_func, T_list, TE_list, xlattice)

x_fit = np.arange(400, 1201, 10)
y_fit = x_fit
z_equilibrium = fitted_function(x_fit, y_fit)

x_fixed = np.ones_like(y_fit) * 1473
z_1473K = fitted_function(x_fixed, y_fit)

x_varying = np.array([884 if y < 884 else y for y in y_fit])
z_varying = fitted_function(x_varying, y_fit)


################################################################################
# Plot CTE evolution.                                                          #
################################################################################

# Start figure.
fig, ax = plt.subplots(figsize=(2.8, 2.6))

from scipy.signal import savgol_filter

# Plot.
T_avg, CTE = compute_CTE(y_fit, z_1473K)
ax.plot(T_avg, CTE, "-", color="#F7931E", label="Low CSRO (1473 K)", linewidth=1.5, alpha=0.9)
T_avg, CTE = compute_CTE(y_fit, z_varying)
ax.plot(T_avg, CTE, "-", color="#37b35f", label="Evolving CSRO", zorder=9, linewidth=1.5, alpha=0.9)
T_avg, CTE = compute_CTE(y_fit, z_equilibrium)
ax.plot(T_avg, CTE, "-", color="#6E96E5", label="Equilibrium CSRO", linewidth=1.5, alpha=0.9)

T1, lat1 = np.loadtxt("data/Francisco_h1.csv", unpack=True)
T1_avg, CTE1 = compute_CTE_nonuniform(T1, lat1)
T1_avg = convolve_1D(T1_avg)
CTE1 = convolve_1D(CTE1)

T2, lat2 = np.loadtxt("data/Francisco_h2.csv", unpack=True)
T2_avg, CTE2 = compute_CTE_nonuniform(T2, lat2)
T2_avg = convolve_1D(T2_avg)
CTE2 = convolve_1D(CTE2)

ax.set_ylim(10.5, 20.5)
ax.set_xlim(350, 1250)

# Add details.
ax.set_ylabel(r"TEC ($10^{-6}\,K^{-1}$)", fontsize=8)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.legend(loc="upper left", fontsize=7)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.yaxis.set_label_coords(-0.09, 0.5)
ax.xaxis.set_label_coords(0.5, -0.08)

# Save figure.
fig.savefig("figures/thermal_expansion.pdf")
plt.close()
