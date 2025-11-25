#!/usr/bin/env python
"""
Generic PIMC Minimization Script for Helium Systems (Aziz 1995 Potential)

This script performs energy minimization on PIMC configurations from worldline files.
System parameters are passed via command line arguments.

Usage:
    python minimize_pimc.py <input_dir> <output_dir> [options]

Arguments:
    input_dir       Directory containing input worldline file (*.dat)
    output_dir      Directory for output files

Options:
    --N             Number of particles (required)
    --box-size      Box size in Angstroms (required)
    --T             Temperature in Kelvin (required)
    --mass          Particle mass (default: 4.0026 for He-4)
    --hbar          Reduced Planck constant (default: 7.638)
    --configs       Configuration indices to minimize (default: 0)
                    Example: --configs 0 1 2

Output files (in output_dir):
    - minimized.wl.dat: minimized worldline configurations
    - minimized.est.dat: initial and final energies
    - conf{i}.log: energy log for configuration i
    - conf{i}.trajectory.dat: minimization trajectory for configuration i
"""
import sys
import os
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='PIMC energy minimization for Helium systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_dir', help='Input directory containing worldline file')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--N', type=int, required=True, help='Number of particles')
    parser.add_argument('--box-size', type=float, required=True, help='Box size in Angstroms')
    parser.add_argument('--T', type=float, required=True, help='Temperature in Kelvin')
    parser.add_argument('--mass', type=float, default=4.0026, help='Particle mass (default: 4.0026 for He-4)')
    parser.add_argument('--hbar', type=float, default=3.4812756477, help='Reduced Planck constant (default: 21.8735/(2*pi) = 3.4812756477)')
    parser.add_argument('--configs', type=int, nargs='+', default=[0], help='Configuration indices to minimize')
    parser.add_argument('--save-every', type=int, default=10, help='Save trajectory every N iterations (default: 10)')
    parser.add_argument('--maxiter', type=int, default=10000, help='Maximum iterations (default: 10000)')
    parser.add_argument('--escape-saddles', action='store_true', help='Enable saddle point escape mechanism (default: False)')
    parser.add_argument('--max-saddle-escapes', type=int, default=5, help='Maximum number of saddle escape attempts (default: 5)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PIMC Energy Minimization: Helium System (Aziz 1995 Potential)")
    print("=" * 70)

    # System parameters
    N = args.N
    T = args.T
    box_size = args.box_size
    mass = args.mass
    hbar = args.hbar
    beta = 1/T
    configs_to_minimize = args.configs

    box = jnp.array([box_size, box_size, box_size])

    print(f"\nSystem Parameters:")
    print(f"  N = {N} particles")
    print(f"  Temperature = {T} K")
    print(f"  Box size = {box_size:.2f} Å")
    print(f"  β = {beta:.4f}, ℏ = {hbar:.4f}, mass = {mass}")

    # Find input worldline file
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.dat')]
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .dat files found in {args.input_dir}")
    if len(input_files) > 1:
        print(f"\nWarning: Multiple .dat files found in {args.input_dir}")
        print(f"Using: {input_files[0]}")

    wlfile = os.path.join(args.input_dir, input_files[0])
    print(f"\nLoading configurations from {wlfile}...")
    paths_dict = load_pimc_worldline_file(wlfile, Lx=box_size, Ly=box_size, Lz=box_size)

    print(f"  Total configurations in file: {len(paths_dict)}")
    print(f"  Configurations to process: {configs_to_minimize}")

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
    classical_energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn, r_cutoff=7.0, r_sw=6.3)
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)

    # Setup output directory and files
    os.makedirs(args.output_dir, exist_ok=True)

    minimized_wl_file = os.path.join(args.output_dir, 'minimized.wl.dat')
    estimator_file = os.path.join(args.output_dir, 'minimized.est.dat')

    print(f"\nOutput files:")
    print(f"  Directory: {args.output_dir}")
    print(f"  Minimized worldline: minimized.wl.dat")
    print(f"  Estimator data: minimized.est.dat")

    # Initialize output files
    wl_handle = open(minimized_wl_file, 'w')
    wl_handle.write("# PIMCID: minimized-worldlines\n")

    est_handle = open(estimator_file, 'w')
    est_handle.write("config,Urp_initial,Urp_final,E_sp_initial,E_sp_final,E_int_initial,E_int_final,E_qm_initial,E_qm_final\n")

    # Loop over configurations
    import time
    total_start_time = time.time()

    for config_idx in configs_to_minimize:
        print(f"\n{'=' * 70}")
        print(f"Processing Configuration {config_idx}")
        print(f"{'=' * 70}")

        original_path = paths_dict[config_idx]

        # Setup per-config output files
        log_file = os.path.join(args.output_dir, f'conf{config_idx}.log')
        trajectory_file = os.path.join(args.output_dir, f'conf{config_idx}.trajectory.dat')

        # Check for resume
        resume_path, resume_iteration = read_last_config_from_trajectory(
            trajectory_file, Lx=box_size, Ly=box_size, Lz=box_size
        )

        resume_mode = resume_path is not None
        if resume_mode:
            print(f"\nResuming from iteration {resume_iteration}")
            path = resume_path
        else:
            print(f"\nStarting fresh minimization")
            path = original_path

        # Calculate energies
        original_initial_result = pimc_energy_fn(original_path, beta, hbar, mass)
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

        # Prepare minimization
        minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
            pimc_energy_fn, path, beta, hbar, mass
        )

        metadata = {
            'config_index': config_idx,
            'N': N,
            'M': M,
            'T': T,
            'beta': beta,
            'hbar': hbar,
            'mass': mass,
            'box_size': box_size
        }

        # Run minimization
        config_start_time = time.time()

        results = find_local_minimum(
            energy_fn=minimization_energy_fn,
            method='trust-ncg',
            xyz_initial=path.beadCoord,
            log_file=log_file,
            log_every=1,
            trajectory_file=trajectory_file,
            trajectory_path_template=path_template,
            save_trajectory_every=args.save_every,
            gtol=1e-6,
            maxiter=args.maxiter,
            maxfun=100000,
            energy_change_tol=1e-4,
            escape_saddles=args.escape_saddles,
            max_saddle_escapes=args.max_saddle_escapes,
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

        # Write outputs
        write_pimc_worldline_config(wl_handle, final_path, config_idx)
        wl_handle.flush()

        est_handle.write(
            f"{config_idx},"
            f"{original_initial_result['Urp']},{final_result['Urp']},"
            f"{original_initial_result['E_sp']},{final_result['E_sp']},"
            f"{original_initial_result['E_int']},{final_result['E_int']},"
            f"{original_initial_result['E_qm']},{final_result['E_qm']}\n"
        )
        est_handle.flush()

    # Close files
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
    print(f"\nOutput files in {args.output_dir}:")
    print(f"  - minimized.wl.dat")
    print(f"  - minimized.est.dat")
    for config_idx in configs_to_minimize:
        print(f"  - conf{config_idx}.log")
        print(f"  - conf{config_idx}.trajectory.dat")


if __name__ == "__main__":
    main()
