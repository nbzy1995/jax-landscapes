"""
Test local minimization routines.
"""

import json
import os
import time
import pytest
import jax
import jax.numpy as jnp
from jax_md import space

from jax_landscape.local_minima import find_local_minimum
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_neighborlist, build_energy_fn_aziz_1995_no_neighborlist
from jax_landscape.pimc_energy import build_pimc_energy_fn, build_pimc_energy_fn_xyz
from jax_landscape.io.pimc import load_pimc_worldline_file, Path

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")

nm_to_Angstrom = 10.0
KJmol_to_KBK = 120.2722922542  # kJ/mol to kB K

def load_test_data(filename):
    """Load test data from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return {
        'xyz': jnp.array(data['xyz']).reshape(-1, 3),
        'xyz_IS': jnp.array(data['xyz_IS']).reshape(-1, 3),
        'box': jnp.array(data['box']),
        'E': data['E'],
        'E_IS': data['E_IS']
    }


@pytest.mark.parametrize("data_file", [
    'tests/test_data/aziz1995-N6-Nbeads1.json',
])
def test_local_minimum_classical_no_neighborlist(data_file):
    """Test local minimization for classical Helium system without neighbor list."""
    # Load test data
    data = load_test_data(data_file)

    xyz_initial = data['xyz'] * nm_to_Angstrom
    box_size = data['box'] * nm_to_Angstrom
    xyz_reference = data['xyz_IS'] * nm_to_Angstrom
    energy_reference = data['E_IS'] * KJmol_to_KBK

    print(f"\nTesting local minimization (no neighbor list) with {xyz_initial.shape[0]} particles:")

    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/N{xyz_initial.shape[0]}_minimization_no_nl.log"

    # Build energy function
    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # Start timing the minimization
    start_time = time.time()

    results = find_local_minimum(
        energy_fn=energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=10
    )

    print(f"Initial energy: {results['energy_initial']:.8f}")
    
    # End timing and calculate duration
    end_time = time.time()
    minimization_time = end_time - start_time

    print(f"Iterations: {results['nit']}, Function evaluations: {results['nfev']}")
    print(f"Minimization time: {minimization_time:.3f} seconds")
    print(f"Performance: {results['nfev']/minimization_time:.1f} function evaluations per second")

    assert results['success'], f"Optimization failed: {results['message']}"

    energy_tolerance = 5e-4  # Relaxed tolerance for different optimization paths
    energy_diff = abs(results['energy_final'] - energy_reference)
    print(f"Final energy: {results['energy_final']:.8f}")
    print(f"Reference energy: {energy_reference:.8f}")

    xyz_final = results['xyz_final']
    max_diff = jnp.max(jnp.abs(xyz_final - xyz_reference))
    xyz_tolerance = 3.0  # Relaxed tolerance as different minima can have similar energies
    print(f"xyz max component difference: {max_diff:.6f}")

    assert results['energy_final'] < results['energy_initial'], \
        "Final energy should be lower than initial energy"

    assert energy_diff < energy_tolerance, \
        f"Compared to the reference energy, the absolute difference {energy_diff:.2e} exceeds tolerance {energy_tolerance:.2e}"

    assert max_diff < xyz_tolerance, \
        f"Compared to reference xyz, the max component difference {max_diff:.6f} exceeds tolerance {xyz_tolerance}"


@pytest.mark.parametrize("data_file", [
    'tests/test_data/aziz1995-N6-Nbeads1.json',
])
def test_local_minimum_classical_with_neighborlist(data_file):
    """Test local minimization for classical Helium system with neighbor list."""
    # Load test data
    data = load_test_data(data_file)

    xyz_initial = data['xyz'] * nm_to_Angstrom
    box_size = data['box'] * nm_to_Angstrom
    xyz_reference = data['xyz_IS'] * nm_to_Angstrom
    energy_reference = data['E_IS'] * KJmol_to_KBK

    print(f"\nTesting local minimization (with neighbor list) with {xyz_initial.shape[0]} particles:")

    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/N{xyz_initial.shape[0]}_minimization_with_nl.log"

    # Build energy function with neighbor list
    displacement_fn, _ = space.periodic(box_size)
    neighbor_fn, energy_fn = build_energy_fn_aziz_1995_neighborlist(displacement_fn, box_size)

    # Start timing the minimization
    start_time = time.time()

    results = find_local_minimum(
        energy_fn=energy_fn,
        xyz_initial=xyz_initial,
        neighbor_fn=neighbor_fn,
        log_file=log_file,
        log_every=10
    )

    print(f"Initial energy: {results['energy_initial']:.8f}")

    # End timing and calculate duration
    end_time = time.time()
    minimization_time = end_time - start_time

    print(f"Iterations: {results['nit']}, Function evaluations: {results['nfev']}")
    print(f"Minimization time: {minimization_time:.3f} seconds")
    print(f"Performance: {results['nfev']/minimization_time:.1f} function evaluations per second")

    assert results['success'], f"Optimization failed: {results['message']}"

    energy_tolerance = 5e-4  # Relaxed tolerance for different optimization paths
    energy_diff = abs(results['energy_final'] - energy_reference)
    print(f"Final energy: {results['energy_final']:.8f}")
    print(f"Reference energy: {energy_reference:.8f}")

    xyz_final = results['xyz_final']
    max_diff = jnp.max(jnp.abs(xyz_final - xyz_reference))
    xyz_tolerance = 3.0  # Relaxed tolerance as different minima can have similar energies
    print(f"xyz max component difference: {max_diff:.6f}")

    assert results['energy_final'] < results['energy_initial'], \
        "Final energy should be lower than initial energy"

    assert energy_diff < energy_tolerance, \
        f"Compared to the reference energy, the absolute difference {energy_diff:.2e} exceeds tolerance {energy_tolerance:.2e}"

    assert max_diff < xyz_tolerance, \
        f"Compared to reference xyz, the max component difference {max_diff:.6f} exceeds tolerance {xyz_tolerance}"

@pytest.mark.parametrize("wl_file", [
    'tests/test_data/N2-Nbeads3-cycle1.dat','tests/test_data/N2-Nbeads3-cycle2.dat'
])
def test_minimize_pimc(wl_file):
    """Test PIMC minimization on N2-Nbeads3 system."""
    # Load PIMC data
    box_size_angstrom = 10.0
    paths_dict = load_pimc_worldline_file(wl_file, Lx=box_size_angstrom, Ly=box_size_angstrom, Lz=box_size_angstrom)

    # Use first configuration
    path = paths_dict[0]

    M, N, _ = path.beadCoord.shape
    print(f"\nTesting PIMC minimization:")
    print(f"  Time slices (M): {M}")
    print(f"  Particles   (N): {N}")
    print(f"  Total beads: {M * N}")

    # Setup energy functions
    displacement_fn, _ = space.periodic(box_size_angstrom)
    classical_energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # PIMC parameters (He-4 at moderate temperature)
    beta = 10.0  # 1/(kB*T) where T ~ 1K
    hbar = 7.638  # ℏ in units consistent with Angstrom and kB*K
    mass = 4.0026  # He-4 mass in atomic units

    # Build PIMC energy and wrap for minimization
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
    minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
        pimc_energy_fn, path, beta, hbar, mass
    )

    xyz_initial = path.beadCoord  # shape (M, N, 3)

    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/pimc_M{M}_N{N}_minimization.log"
    trajectory_file = f"tests/tmp/pimc_M{M}_N{N}_trajectory.dat"

    # Run minimization
    start_time = time.time()

    results = find_local_minimum(
        energy_fn=minimization_energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=10,
        trajectory_file=trajectory_file,
        trajectory_path_template=path_template,
        save_trajectory_every=5,
        gtol=1e-6,
        maxiter=10000
    )

    end_time = time.time()
    minimization_time = end_time - start_time

    print(f"Initial energy (Urp): {results['energy_initial']:.8f}")
    print(f"Final energy (Urp): {results['energy_final']:.8f}")
    print(f"Energy reduction: {results['energy_initial'] - results['energy_final']:.8f}")
    print(f"Iterations: {results['nit']}, Function evaluations: {results['nfev']}")
    print(f"Minimization time: {minimization_time:.3f} seconds")

    assert results['success'], f"Optimization failed: {results['message']}"
    assert results['xyz_final'].shape == xyz_initial.shape, "Output shape should match input"
    assert results['energy_final'] < results['energy_initial'], \
        "Final energy should be lower than initial energy"

    # Test trajectory file can be read back
    print(f"\nVerifying trajectory file...")
    trajectory_paths = load_pimc_worldline_file(trajectory_file, Lx=box_size_angstrom, Ly=box_size_angstrom, Lz=box_size_angstrom)
    num_configs = len(trajectory_paths)
    print(f"  Trajectory contains {num_configs} configurations")

    assert num_configs > 0, "Trajectory file should contain at least one configuration"

    # Verify we can access a Path from the trajectory
    first_config_key = list(trajectory_paths.keys())[0]
    first_config_path = trajectory_paths[first_config_key]
    assert first_config_path.beadCoord.shape == xyz_initial.shape, "Trajectory config should have same shape as initial"

    print(f"  Trajectory file verified successfully!")

