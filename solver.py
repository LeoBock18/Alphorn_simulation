import numpy as np
from scipy.signal import find_peaks
import re
import os
import warnings
import matplotlib.pyplot as plt

from mpi4py import MPI

from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl


def solve_helmholtz(mesh_filename: str, freqs: np.ndarray, u_n: float, R: float, results_dir: str = "results", verbose: bool = False):
    """
    Solves the Helmholtz problem on the alphorn mesh given in input and calculates the input impedance for various freqs.
    Real parts, Imaginary parts and Moduli are stored in a CSV file with the same labeling name as mesh file.

    :param mesh_filename: Name of the file where the mesh is stored.
    :type mesh_filename: str
    :param freqs: Range of frequencies for which to solve Helmholtz [Hz].
    :type freqs: ndarray
    :param u_n: Input velocity [mm/s].
    :type u_n: float
    :param R: Radius of hemispherical part (must be coherent with the given mesh) [mm].
    :type R: float
    :param result_dir: Folder where to save results.
    :type R: str
    :param verbose: Displays progress in computation
    :type verbose: bool
    """

    # Load the provided mesh
    domain, _, facet_tags, _, _, _ = io.gmshio.read_from_msh(mesh_filename, MPI.COMM_WORLD, 0, gdim=3)

    # Define space
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define ds and dx for integration
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)

    # Parameters
    rho0 = 1.225*1000/10**9  # g/mm^3
    c = 340.0*1000  # mm/s

    # k changes with freq: define it as fem.Constant
    k = fem.Constant(domain, default_scalar_type(0))


    # Define bilinear forms
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (
        ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
        - (k*R - 1j)/R * ufl.inner(p, v) * ds(3)
        - k**2 * ufl.inner(p, v) * dx
    )
    L = 1j * c * rho0 * k * ufl.inner(u_n, v) * ds(2)

    p_a = fem.Function(V)
    p_a.name = "pressure"

    # Define problem
    problem = LinearProblem(
        a,
        L,
        u=p_a,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        form_compiler_options={"scalar_type": "complex"},
    )

    # Define area and pressure at the inlet (symbolic form)
    area_form = fem.form(1.0 * ds(2))
    pressure_form = fem.form(p_a * ds(2))

    # Define arrays to store impedance values
    Z_in_real = []
    Z_in_imag = []
    Z_moduli = []

    for i, f in enumerate(freqs):
        # Show optional loop progress
        if verbose:
            print(f"Processing item {i + 1} of {len(freqs)}")

        omega = 2 * np.pi * f
        k.value = omega / c
        
        problem.solve()

        # Integrate pressure over inlet
        p_integral = fem.assemble_scalar(pressure_form)
        p_integral = domain.comm.allreduce(p_integral, op=MPI.SUM)

        # Integrate area over inlet
        inlet_area = fem.assemble_scalar(area_form)
        inlet_area = domain.comm.allreduce(inlet_area, op=MPI.SUM)

        # Compute average pressure at inlet
        Z_in = p_integral / (u_n*inlet_area)

        Z_in_real.append(Z_in.real)
        Z_in_imag.append(Z_in.imag)
        Z_moduli.append(abs(Z_in))
    
    # Stack results in a single array
    results = np.column_stack((Z_in_real, Z_in_imag, Z_moduli))
    
    # Extract number of the file in input and use it to construct output file name
    basename = os.path.basename(mesh_filename)
    match = re.search(r'(\d+)\.msh$', basename)
    if not match:
        warnings.warn(f"No trailing number found before '.msh' in filename: {mesh_filename}. Automatically set number to 0")
        number = 0
    else:
        number = match.group(1)

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Construct the result filename inside directory
    result_filename = f"result{number}.csv"
    result_path = os.path.join(results_dir, result_filename)

    # Save results to CSV
    np.savetxt(result_path, results, delimiter=",", header="Real Z, Imag Z, Modulus Z", comments='', fmt="%.3f")


def compute_resonance_freqs(results_path: str, freqs: np.ndarray):
    """
    Computes resonance frequencies.

    :param results_path: Path to results.
    :type R: str
    :param freqs: Range of frequencies [Hz].
    :type freqs: ndarray
    :return: Resonance frequencies [Hz].
    :rtype: ndarray
    """  

    # Load data from file
    data = np.loadtxt(results_path, delimiter=",", skiprows=1)
    Z_moduli = data[:, 2]

    # Find indeces of peaks in input impedance moduli and use them to extract frequencies
    peaks, _ = find_peaks(Z_moduli)
    resonance_freqs = freqs[peaks]

    return resonance_freqs



def plot_results(results_path: str, freqs: np.ndarray, plots_dir: str = "plots"):
    """
    Stores curves for input impedance inside a file with the same labeling name as results file.

    :param results_path: Path to results.
    :type R: str
    :param freqs: Range of frequencies [Hz] related to results data.
    :type freqs: ndarray
    :param results_path: Folder where to store plots.
    :type R: str
    """

    # Load data from file
    data = np.loadtxt(results_path, delimiter=",", skiprows=1)
    Z_in_real = data[:, 0]
    Z_in_imag = data[:, 1]
    Z_moduli = data[:, 2]

    # First subplot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(freqs, Z_in_real, label="Re(Z_in)")
    axs[0].plot(freqs, Z_in_imag, label="Im(Z_in)")
    axs[0].set_title('Real and Imaginary parts')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot
    axs[1].plot(freqs, Z_moduli)
    axs[1].set_title('Moduli')
    axs[1].grid(True)

    # Main title for the whole figure
    a = freqs[0]
    b = freqs[-1]
    fig.suptitle(f'Interval [{a}, {b}] Hz', fontsize=14)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    # Extract number of the file in input and use it to construct output file name
    basename = os.path.basename(results_path)
    match = re.search(r'(\d+)\.csv$', basename)
    if not match:
        warnings.warn(f"No trailing number found before '.msh' in filename: {results_path}. Automatically set number to 0")
        number = 0
    else:
        number = match.group(1)

    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Construct the plot filename inside directory
    plot_filename = f"plot{number}.png"
    plot_path = os.path.join(plots_dir, plot_filename)

    # Save the plot to a file
    plt.savefig(plot_path)


def main():

    mesh_filename = "alphorn_meshes/mesh23.msh"
    freqs = np.arange(281, 350, 1)
    R = 447.09 #mm
    u_n = 1.0*1000 # mm/s

    solve_helmholtz(mesh_filename, freqs, u_n, R, verbose = True)
    plot_results("results/result23.csv", freqs)


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()