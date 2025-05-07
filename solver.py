import numpy as np
from scipy.signal import find_peaks

from mpi4py import MPI

from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl


def solve_helmholtz(filename: str, freqs: ndarray, u_n: float, R: float):
    """
    Solves the Helmholtz problem on the alphorn mesh given in input and calculate the resonance frequencies
        Args:
        filename (str): name of the file where mesh is stored
        freqs (ndarray): range of frequencies where to solve Helmholtz
        u_n (float): input velocity
        R (float): radius of hemispherical part (MUST BE COHERENT WITH THE GIVEN MESH)
    """

    # Load the provided mesh
    domain, _, facet_tags, _, _, _ = io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, 0, gdim=3)

    # Define space
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define ds and dx for integration
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)

    # Parameters
    rho0 = 1.225*1000/10**9  # g/mm^3
    c = 340.0*1000  # mm/s
    u_n = 1.0*1000 # mm/s
    R = 447.09 # mm

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

    for f in freqs:
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
        p_avg = p_integral / inlet_area

        # Since u_n = 1 m/s, Z_in = p_avg / u_n = p_avg
        Z_in_real.append(p_avg.real)
        Z_in_imag.append(p_avg.imag)
        Z_moduli.append(abs(p_avg))
    
    peaks, _ = find_peaks(Z_moduli)
    # resonance_freqs = [(i, Z_mod[i]) for i in peaks]
    resonance_freqs = peaks + freqs[0]
    resonance_freqs