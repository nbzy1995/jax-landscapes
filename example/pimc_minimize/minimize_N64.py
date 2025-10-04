#!/usr/bin/env python
"""
PIMC Minimization Example: N64 Helium System

This script performs energy minimization on PIMC configurations from a worldline file.
The system parameters are taken from test_full_wl in tests/test_pimc_energy.py.

Output format:
- Per-configuration logging:
    - Log file: N64.confX.log (energy at each iteration)
    - Trajectory file: N64.confX.trajectory.dat (minimization path)
- Final results (written incrementally):
    - Worldline file: N64.minimized.wl.dat (all minimized configurations)
    - Estimator file: N64.minimized.est.dat (initial and final energies)

Usage:
    python minimize_N64.py [config_indices...]

    config_indices: Optional list of configuration indices to minimize (default: 0)
    Example: python minimize_N64.py 0 1 2  # minimizes first 3 configurations
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../..'))

from jax_landscape.io.pimc import (
    load_pimc_worldline_file,
    write_pimc_worldline_config,
    read_last_config_from_trajectory
)
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

    # Parse command line arguments for configuration indices
    if len(sys.argv) > 1:
        configs_to_minimize = [int(arg) for arg in sys.argv[1:]]
    else:
        configs_to_minimize = [0]  # Default: minimize first configuration

    print(f"\nConfigurations to minimize: {configs_to_minimize}")

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

    # Load PIMC configurations
    wlfile = 'N64.dat'
    print(f"\nLoading configurations from {wlfile}...")
    paths_dict = load_pimc_worldline_file(wlfile, Lx=L, Ly=L, Lz=L)

    print(f"  Total configurations in file: {len(paths_dict)}")
    print(f"  Configurations to process: {len(configs_to_minimize)}")

    # Validate configuration indices
    for cfg_idx in configs_to_minimize:
        if cfg_idx not in paths_dict:
            raise ValueError(f"Configuration {cfg_idx} not found in {wlfile}")

    M = paths_dict[configs_to_minimize[0]].numTimeSlices
    print(f"  Time slices (M) = {M}")
    print(f"  Total beads per config = {M * N}")

    # Build energy functions
    print(f"\nBuilding energy functions...")
    displacement_fn, _ = space.periodic(box)
    classical_energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)

    # Setup output files
    base_name = wlfile.replace('.dat', '')
    minimized_wl_file = f'{base_name}.minimized.wl.dat'
    estimator_file = f'{base_name}.minimized.est.dat'

    print(f"\nInitializing output files...")
    print(f"  Minimized worldline: {minimized_wl_file}")
    print(f"  Estimator data: {estimator_file}")

    # Initialize worldline file with header
    wl_handle = open(minimized_wl_file, 'w')
    wl_handle.write("# PIMCID: minimized-worldlines\n")

    # Initialize estimator file with header
    est_handle = open(estimator_file, 'w')
    est_handle.write("config,Urp_initial,Urp_final,E_sp_initial,E_sp_final,E_int_initial,E_int_final,E_qm_initial,E_qm_final\n")

    # Loop over configurations to minimize
    import time
    total_start_time = time.time()

    for config_idx in configs_to_minimize:
        print(f"\n{'=' * 70}")
        print(f"Processing Configuration {config_idx}")
        print(f"{'=' * 70}")

        # Get configuration
        original_path = paths_dict[config_idx]

        # Setup per-config output files
        log_file = f'{base_name}.conf{config_idx}.log'
        trajectory_file = f'{base_name}.conf{config_idx}.trajectory.dat'

        # Check for resume
        resume_path, resume_iteration = read_last_config_from_trajectory(
            trajectory_file, Lx=L, Ly=L, Lz=L
        )

        resume_mode = resume_path is not None
        if resume_mode:
            print(f"\nResuming from iteration {resume_iteration}")
            print(f"  Found existing trajectory file: {trajectory_file}")
            # Use the resumed path for minimization
            path = resume_path
        else:
            print(f"\nStarting fresh minimization")
            path = original_path

        # Calculate true initial energy (always from original configuration)
        original_initial_result = pimc_energy_fn(original_path, beta, hbar, mass)

        # Calculate current energy (from resume point if resuming)
        current_result = pimc_energy_fn(path, beta, hbar, mass)

        if resume_mode:
            print(f"\nOriginal Initial Energies:")
            print(f"  Urp (total) = {original_initial_result['Urp']:.2f} kB·K")
            print(f"\nCurrent Energies (iteration {resume_iteration}):")
        else:
            print(f"\nInitial Energies:")
        print(f"  Urp (total) = {current_result['Urp']:.2f} kB·K")
        print(f"  E_sp (spring) = {current_result['E_sp']:.2f} kB·K")
        print(f"  E_int (interaction) = {current_result['E_int']:.2f} kB·K")
        print(f"  E_qm (quantum) = {current_result['E_qm']:.2f} kB·K")

        # Prepare minimization energy function
        minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
            pimc_energy_fn, path, beta, hbar, mass
        )

        print(f"\nMinimization settings:")
        print(f"  Log file: {log_file}")
        print(f"  Trajectory file: {trajectory_file}")
        print(f"  Saving trajectory every 50 iterations")

        # Prepare metadata for log file
        metadata = {
            'config_index': config_idx,
            'N': N,
            'M': M,
            'T': T,
            'beta': beta,
            'hbar': hbar,
            'mass': mass,
            'L': L,
            'density': n
        }

        # Run minimization
        config_start_time = time.time()

        results = find_local_minimum(
            energy_fn=minimization_energy_fn,
            xyz_initial=path.beadCoord,
            log_file=log_file,
            log_every=100,
            trajectory_file=trajectory_file,
            trajectory_path_template=path_template,
            save_trajectory_every=100,
            gtol=1e-3,
            maxiter=10000,
            energy_change_tol=1e-2,
            initial_iteration=resume_iteration,
            resume_mode=resume_mode,
            metadata=metadata
        )

        config_end_time = time.time()
        minimization_time = config_end_time - config_start_time

        # Report results
        print(f"\nMinimization Complete!")
        print(f"  Success: {results['success']}")
        print(f"  Message: {results['message']}")
        print(f"  Iterations: {results['nit']}")
        print(f"  Function evaluations: {results['nfev']}")
        print(f"  Time: {minimization_time:.1f} seconds ({results['nfev']/minimization_time:.1f} evals/sec)")

        # Calculate final energies
        xyz_final = results['xyz_final'].reshape(path.beadCoord.shape)
        class FinalPath:
            def __init__(self):
                self.beadCoord = xyz_final
                self.next = path.next
                self.prev = path.prev
                self.wlIndex = path.wlIndex
                self.write_order = path.write_order

        final_path = FinalPath()
        final_result = pimc_energy_fn(final_path, beta, hbar, mass)

        print(f"\nFinal Energies:")
        print(f"  Urp (total) = {final_result['Urp']:.2f} kB·K")
        print(f"  E_sp (spring) = {final_result['E_sp']:.2f} kB·K")
        print(f"  E_int (interaction) = {final_result['E_int']:.2f} kB·K")
        print(f"  E_qm (quantum) = {final_result['E_qm']:.2f} kB·K")

        print(f"\nEnergy Reduction:")
        print(f"  Urp: {original_initial_result['Urp'] - final_result['Urp']:.2f} kB·K")

        # Write minimized configuration to worldline file
        print(f"\nWriting minimized configuration to {minimized_wl_file}...")
        write_pimc_worldline_config(wl_handle, final_path, config_idx)
        wl_handle.flush()

        # Write estimator row (always use original initial energies)
        print(f"Writing estimator data to {estimator_file}...")
        est_handle.write(
            f"{config_idx},"
            f"{original_initial_result['Urp']},{final_result['Urp']},"
            f"{original_initial_result['E_sp']},{final_result['E_sp']},"
            f"{original_initial_result['E_int']},{final_result['E_int']},"
            f"{original_initial_result['E_qm']},{final_result['E_qm']}\n"
        )
        est_handle.flush()

    # Close output files
    wl_handle.close()
    est_handle.close()

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Final summary
    print(f"\n{'=' * 70}")
    print("All Minimizations Complete!")
    print(f"{'=' * 70}")
    print(f"Total configurations processed: {len(configs_to_minimize)}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"\nOutput files created:")
    print(f"  - {minimized_wl_file} (minimized worldline configurations)")
    print(f"  - {estimator_file} (initial and final energies)")
    for config_idx in configs_to_minimize:
        print(f"  - {base_name}.conf{config_idx}.log (iteration log)")
        print(f"  - {base_name}.conf{config_idx}.trajectory.dat (minimization trajectory)")

if __name__ == "__main__":
    main()
