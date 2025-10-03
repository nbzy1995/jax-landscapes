#!/usr/bin/env python
"""
PIMC Minimization Example: N64 Helium System

This script performs energy minimization on a 64-particle Helium PIMC configuration
and saves the trajectory for visualization. The system parameters are taken from
test_full_wl in tests/test_pimc_energy.py.


TODO: change the output pattern:
- The input file is a worldline file that can contain multiple configurations
- We should specifiy which configuration to minimize (default: the first one)
- For each configuration that we minimize, we output the logging info:
    - A log file with the energy at each iteration.
    - A trajectory file that contains the minimization path for that configuration. The file also contains the metadata of that config. 
    - the filename should reflect which configuration it is (e.g., conf0, conf1, etc.)
- For the result, we should output:
    - one worldline file that contains all the minimized configurations, with configuration step same as input file.
    - one estimator file that contains the initial and final energies for each configuration, of the same format as the estimator file output by the PIMC simulation.


Usage:
    python minimize_N64.py
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../..'))

from jax_landscape.io.pimc import load_pimc_worldline_file
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_no_neighborlist
from jax_landscape.pimc_energy import build_pimc_energy_fn, build_pimc_energy_fn_xyz
from jax_landscape.local_minima import find_local_minimum

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")

def main():
    print("=" * 70)
    print("PIMC Energy Minimization: N=64 Helium System")
    print("=" * 70)

    # System parameters (from test_full_wl)
    N = 64                    # Number of particles
    n = 0.0218               # Density in Angstrom^-3
    T = 1.55                 # Temperature [K]
    beta = 1/T               # Inverse temperature (reduced units)
    hbar = 21.8735/(2*np.pi) # Reduced Planck constant
    mass = 1.0               # Helium mass (reduced units)

    L = (N/n)**(1/3)         # Box length in Angstrom
    box = jnp.array([L, L, L])

    print(f"\nSystem Parameters:")
    print(f"  N = {N} particles")
    print(f"  Density = {n:.4f} Å⁻³")
    print(f"  Temperature = {T} K")
    print(f"  Box size = {L:.2f} Å")
    print(f"  β = {beta:.4f}, ℏ = {hbar:.4f}, mass = {mass}")

    # Load PIMC configuration
    wlfile = 'N64.dat'
    print(f"\nLoading configuration from {wlfile}...")
    paths_dict = load_pimc_worldline_file(wlfile, Lx=L, Ly=L, Lz=L)
    path = paths_dict[0]
    M = path.numTimeSlices

    print(f"  Time slices (M) = {M}")
    print(f"  Total beads = {M * N}")
    print(f"  Configuration shape = {path.beadCoord.shape}")

    # Build energy functions
    print(f"\nBuilding energy functions...")
    displacement_fn, _ = space.periodic(box)
    classical_energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)

    # Calculate initial energy
    initial_result = pimc_energy_fn(path, beta, hbar, mass)
    print(f"\nInitial Energies:")
    print(f"  Urp (total) = {initial_result['Urp']:.2f} kB·K")
    print(f"  E_sp (spring) = {initial_result['E_sp']:.2f} kB·K")
    print(f"  E_int (interaction) = {initial_result['E_int']:.2f} kB·K")
    print(f"  E_qm (quantum) = {initial_result['E_qm']:.2f} kB·K")

    # Prepare minimization energy function
    minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
        pimc_energy_fn, path, beta, hbar, mass
    )

    # Setup output files
    trajectory_file = 'N64.conf0.minimization.wl.dat'
    log_file = 'N64.conf0.minimization.log'

    print(f"\nStarting minimization...")
    print(f"  Log file: {log_file}")
    print(f"  Trajectory file: {trajectory_file}")
    print(f"  Saving trajectory every 50 iterations")

    # Run minimization
    import time
    start_time = time.time()

    results = find_local_minimum(
        energy_fn=minimization_energy_fn,
        xyz_initial=path.beadCoord,
        log_file=log_file,
        log_every=50,
        trajectory_file=trajectory_file,
        trajectory_path_template=path_template,
        save_trajectory_every=50,
        gtol=1e-6,
        maxiter=10000
    )

    end_time = time.time()
    minimization_time = end_time - start_time

    # Report results
    print(f"\n" + "=" * 70)
    print("Minimization Complete!")
    print("=" * 70)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Iterations: {results['nit']}")
    print(f"Function evaluations: {results['nfev']}")
    print(f"Time: {minimization_time:.1f} seconds ({results['nfev']/minimization_time:.1f} evals/sec)")

    print(f"\nEnergy Change:")
    print(f"  Initial Urp: {results['energy_initial']:.2f} kB·K")
    print(f"  Final Urp: {results['energy_final']:.2f} kB·K")
    print(f"  Reduction: {results['energy_initial'] - results['energy_final']:.2f} kB·K")

    # Calculate final energies
    xyz_final = results['xyz_final'].reshape(path.beadCoord.shape)
    class FinalPath:
        def __init__(self):
            self.beadCoord = xyz_final
            self.next = path.next
    final_path = FinalPath()
    final_result = pimc_energy_fn(final_path, beta, hbar, mass)

    print(f"\nFinal Energies:")
    print(f"  Urp (total) = {final_result['Urp']:.2f} kB·K")
    print(f"  E_sp (spring) = {final_result['E_sp']:.2f} kB·K")
    print(f"  E_int (interaction) = {final_result['E_int']:.2f} kB·K")
    print(f"  E_qm (quantum) = {final_result['E_qm']:.2f} kB·K")

    print(f"\n✅ Output files created:")
    print(f"   - {log_file} (iteration log with Urp, E_sp, E_int)")
    print(f"   - {trajectory_file} (minimization trajectory)")
    print(f"\n💡 Next: Visualize with visualize_N64_minimization.ipynb")

if __name__ == "__main__":
    main()
