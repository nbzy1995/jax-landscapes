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
from jax_landscape.hessian_eigenvals import compute_hessian_eigenvalues

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


# ============================================================================
# Category A: Property-Based Tests (No Reference Data Required)
# ============================================================================

@pytest.mark.parametrize("data_file", [
    'tests/test_data/aziz1995-N6-Nbeads1.json',
])
def test_local_minimum_properties_classical_no_neighborlist(data_file):
    """
    Rigorous test of local minimum properties without relying on reference data.

    Tests 4 mathematical properties:
    1. Zero gradient: ||grad(E)|| ≈ 0
    2. Energy decrease: E_final < E_initial
    3. Local stability: E(x + δ) ≥ E(x) for random perturbations δ
    4. Hessian positive semi-definite: all eigenvalues ≥ 0
    """
    # Load test data and setup
    data = load_test_data(data_file)
    xyz_initial = data['xyz'] * nm_to_Angstrom
    box_size = data['box'] * nm_to_Angstrom

    print(f"\n{'='*70}")
    print(f"Property-Based Test: Classical System (No Neighbor List)")
    print(f"N = {xyz_initial.shape[0]} particles")
    print(f"{'='*70}")

    # Build energy function
    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # Run minimization
    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/properties_classical_N{xyz_initial.shape[0]}.log"

    results = find_local_minimum(
        energy_fn=energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=10,
        gtol=1e-6,
        energy_change_tol=1e-8
    )

    print(f"\nMinimization completed:")
    print(f"  Initial energy: {results['energy_initial']:.8f}")
    print(f"  Final energy:   {results['energy_final']:.8f}")
    print(f"  Energy change:  {results['energy_initial'] - results['energy_final']:.8f}")
    print(f"  Iterations: {results['nit']}, Function evals: {results['nfev']}")

    assert results['success'], f"Optimization failed: {results['message']}"

    xyz_final = results['xyz_final']

    # ========================================================================
    # Property 1: Zero Gradient Condition
    # ========================================================================
    print(f"\n--- Property 1: Zero Gradient ---")

    grad_fn = jax.grad(energy_fn)
    gradient = grad_fn(xyz_final)
    grad_norm = jnp.linalg.norm(gradient)

    print(f"  Gradient norm: {grad_norm:.6e}")

    grad_tolerance = 1e-5  # Slightly more relaxed than minimization gtol=1e-6
    assert grad_norm < grad_tolerance, \
        f"Gradient norm {grad_norm:.6e} exceeds tolerance {grad_tolerance:.6e}"
    print(f"  ✓ Gradient norm < {grad_tolerance:.6e}")

    # ========================================================================
    # Property 2: Energy Decrease
    # ========================================================================
    print(f"\n--- Property 2: Energy Decrease ---")

    energy_decreased = results['energy_final'] < results['energy_initial']
    energy_decrease = results['energy_initial'] - results['energy_final']

    print(f"  Energy decrease: {energy_decrease:.8f}")

    assert energy_decreased, "Final energy should be lower than initial energy"
    print(f"  ✓ Energy decreased")

    # ========================================================================
    # Property 3: Local Stability (Random Perturbations)
    # ========================================================================
    print(f"\n--- Property 3: Local Stability (Perturbation Test) ---")

    # Test multiple perturbation magnitudes
    perturbation_magnitudes = [0.01, 0.05, 0.1]  # Angstroms
    n_perturbations_per_magnitude = 20

    key = jax.random.PRNGKey(42)
    energy_final_value = results['energy_final']

    # Small tolerance for numerical noise
    stability_tolerance = 1e-6

    total_tests = 0
    total_passed = 0

    for magnitude in perturbation_magnitudes:
        passed = 0
        for i in range(n_perturbations_per_magnitude):
            key, subkey = jax.random.split(key)

            # Generate random perturbation
            perturbation = jax.random.normal(subkey, shape=xyz_final.shape) * magnitude
            xyz_perturbed = xyz_final + perturbation

            # Compute energy of perturbed configuration
            energy_perturbed = energy_fn(xyz_perturbed)

            # Check if energy increased (or stayed same within tolerance)
            if energy_perturbed >= energy_final_value - stability_tolerance:
                passed += 1

            total_tests += 1
            total_passed += (energy_perturbed >= energy_final_value - stability_tolerance)

        print(f"  Magnitude {magnitude:.3f} Å: {passed}/{n_perturbations_per_magnitude} perturbations have E ≥ E_final")

    success_rate = total_passed / total_tests
    print(f"  Overall: {total_passed}/{total_tests} tests passed ({success_rate*100:.1f}%)")

    # Require at least 95% of perturbations to increase energy
    assert success_rate >= 0.95, \
        f"Only {success_rate*100:.1f}% of perturbations increased energy (expected ≥95%)"
    print(f"  ✓ Local stability verified (≥95% perturbations increase energy)")

    # ========================================================================
    # Property 4: Hessian Positive Semi-Definite
    # ========================================================================
    print(f"\n--- Property 4: Hessian Positive Semi-Definite ---")

    hessian_result = compute_hessian_eigenvalues(
        energy_fn,
        xyz_final,
        return_hessian=False,
        return_eigenvectors=False
    )

    eigenvalues = hessian_result['eigenvalues']

    # Count near-zero eigenvalues (expected: ~3 for translational modes in periodic box)
    near_zero_threshold = 1e-6
    n_near_zero = jnp.sum(jnp.abs(eigenvalues) < near_zero_threshold)

    # Check all eigenvalues are non-negative (allowing small numerical error)
    eigenvalue_tolerance = -1e-7  # Tolerance for numerical precision
    min_eigenvalue = jnp.min(eigenvalues)
    all_positive = jnp.all(eigenvalues >= eigenvalue_tolerance)

    print(f"  Total eigenvalues: {len(eigenvalues)}")
    print(f"  Near-zero eigenvalues (|λ| < {near_zero_threshold:.0e}): {n_near_zero}")
    print(f"  Min eigenvalue: {min_eigenvalue:.6e}")
    print(f"  Max eigenvalue: {jnp.max(eigenvalues):.6e}")

    assert all_positive, \
        f"Found negative eigenvalue {min_eigenvalue:.6e} below tolerance {eigenvalue_tolerance:.6e}"
    print(f"  ✓ All eigenvalues ≥ {eigenvalue_tolerance:.6e} (positive semi-definite)")

    print(f"\n{'='*70}")
    print(f"All 4 properties verified ✓")
    print(f"{'='*70}")


@pytest.mark.parametrize("wl_file", [
    'tests/test_data/N2-Nbeads3-cycle1.dat',
])
def test_local_minimum_properties_pimc(wl_file):
    """
    Rigorous test of local minimum properties for PIMC without reference data.

    Tests 4 mathematical properties:
    1. Zero gradient: ||grad(U_RP)|| ≈ 0
    2. Energy decrease: U_RP_final < U_RP_initial
    3. Local stability: U_RP(x + δ) ≥ U_RP(x) for random perturbations δ
    4. Hessian positive semi-definite: all eigenvalues ≥ 0
    """
    # Load PIMC data
    box_size_angstrom = 10.0
    paths_dict = load_pimc_worldline_file(wl_file, Lx=box_size_angstrom, Ly=box_size_angstrom, Lz=box_size_angstrom)
    path = paths_dict[0]

    M, N, _ = path.beadCoord.shape

    print(f"\n{'='*70}")
    print(f"Property-Based Test: PIMC System")
    print(f"Time slices (M): {M}, Particles (N): {N}, Total beads: {M*N}")
    print(f"{'='*70}")

    # Setup energy functions
    displacement_fn, _ = space.periodic(box_size_angstrom)
    classical_energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # PIMC parameters
    beta = 10.0
    hbar = 7.638
    mass = 4.0026

    # Build PIMC energy
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
    minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
        pimc_energy_fn, path, beta, hbar, mass
    )

    xyz_initial = path.beadCoord

    # Run minimization
    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/properties_pimc_M{M}_N{N}.log"

    results = find_local_minimum(
        energy_fn=minimization_energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=10,
        gtol=1e-6,
        energy_change_tol=1e-8,
        maxiter=10000
    )

    print(f"\nMinimization completed:")
    print(f"  Initial energy (U_RP): {results['energy_initial']:.8f}")
    print(f"  Final energy (U_RP):   {results['energy_final']:.8f}")
    print(f"  Energy change:         {results['energy_initial'] - results['energy_final']:.8f}")
    print(f"  Iterations: {results['nit']}, Function evals: {results['nfev']}")

    assert results['success'], f"Optimization failed: {results['message']}"

    xyz_final = results['xyz_final']

    # ========================================================================
    # Property 1: Zero Gradient Condition
    # ========================================================================
    print(f"\n--- Property 1: Zero Gradient ---")

    grad_fn = jax.grad(lambda xyz: minimization_energy_fn(xyz)['energy'])
    gradient = grad_fn(xyz_final)
    grad_norm = jnp.linalg.norm(gradient)

    print(f"  Gradient norm: {grad_norm:.6e}")

    grad_tolerance = 1e-5  # Slightly more relaxed than minimization gtol=1e-6
    assert grad_norm < grad_tolerance, \
        f"Gradient norm {grad_norm:.6e} exceeds tolerance {grad_tolerance:.6e}"
    print(f"  ✓ Gradient norm < {grad_tolerance:.6e}")

    # ========================================================================
    # Property 2: Energy Decrease
    # ========================================================================
    print(f"\n--- Property 2: Energy Decrease ---")

    energy_decreased = results['energy_final'] < results['energy_initial']
    energy_decrease = results['energy_initial'] - results['energy_final']

    print(f"  Energy decrease: {energy_decrease:.8f}")

    assert energy_decreased, "Final energy should be lower than initial energy"
    print(f"  ✓ Energy decreased")

    # ========================================================================
    # Property 3: Local Stability (Random Perturbations)
    # ========================================================================
    print(f"\n--- Property 3: Local Stability (Perturbation Test) ---")

    perturbation_magnitudes = [0.01, 0.05, 0.1]
    n_perturbations_per_magnitude = 20

    key = jax.random.PRNGKey(42)
    energy_final_value = results['energy_final']
    stability_tolerance = 1e-6

    total_tests = 0
    total_passed = 0

    for magnitude in perturbation_magnitudes:
        passed = 0
        for i in range(n_perturbations_per_magnitude):
            key, subkey = jax.random.split(key)

            perturbation = jax.random.normal(subkey, shape=xyz_final.shape) * magnitude
            xyz_perturbed = xyz_final + perturbation

            energy_result = minimization_energy_fn(xyz_perturbed)
            energy_perturbed = energy_result['energy']

            if energy_perturbed >= energy_final_value - stability_tolerance:
                passed += 1

            total_tests += 1
            total_passed += (energy_perturbed >= energy_final_value - stability_tolerance)

        print(f"  Magnitude {magnitude:.3f} Å: {passed}/{n_perturbations_per_magnitude} perturbations have E ≥ E_final")

    success_rate = total_passed / total_tests
    print(f"  Overall: {total_passed}/{total_tests} tests passed ({success_rate*100:.1f}%)")

    assert success_rate >= 0.95, \
        f"Only {success_rate*100:.1f}% of perturbations increased energy (expected ≥95%)"
    print(f"  ✓ Local stability verified (≥95% perturbations increase energy)")

    # ========================================================================
    # Property 4: Hessian Positive Semi-Definite
    # ========================================================================
    print(f"\n--- Property 4: Hessian Positive Semi-Definite ---")

    hessian_result = compute_hessian_eigenvalues(
        minimization_energy_fn,
        xyz_final,
        return_hessian=False,
        return_eigenvectors=False
    )

    eigenvalues = hessian_result['eigenvalues']

    near_zero_threshold = 1e-6
    n_near_zero = jnp.sum(jnp.abs(eigenvalues) < near_zero_threshold)

    eigenvalue_tolerance = -1e-7  # Tolerance for numerical precision
    min_eigenvalue = jnp.min(eigenvalues)
    all_positive = jnp.all(eigenvalues >= eigenvalue_tolerance)

    print(f"  Total eigenvalues: {len(eigenvalues)}")
    print(f"  Near-zero eigenvalues (|λ| < {near_zero_threshold:.0e}): {n_near_zero}")
    print(f"  Min eigenvalue: {min_eigenvalue:.6e}")
    print(f"  Max eigenvalue: {jnp.max(eigenvalues):.6e}")

    assert all_positive, \
        f"Found negative eigenvalue {min_eigenvalue:.6e} below tolerance {eigenvalue_tolerance:.6e}"
    print(f"  ✓ All eigenvalues ≥ {eigenvalue_tolerance:.6e} (positive semi-definite)")

    print(f"\n{'='*70}")
    print(f"All 4 properties verified ✓")
    print(f"{'='*70}")


# ============================================================================
# Category B: Reference Data Tests
# ============================================================================

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
        log_every=1
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


# @pytest.mark.parametrize("data_file", [
#     'tests/test_data/aziz1995-N6-Nbeads1.json',
# ])
# def test_local_minimum_classical_with_neighborlist(data_file):
#     """Test local minimization for classical Helium system with neighbor list."""
#     # Load test data
#     data = load_test_data(data_file)

#     xyz_initial = data['xyz'] * nm_to_Angstrom
#     box_size = data['box'] * nm_to_Angstrom
#     xyz_reference = data['xyz_IS'] * nm_to_Angstrom
#     energy_reference = data['E_IS'] * KJmol_to_KBK

#     print(f"\nTesting local minimization (with neighbor list) with {xyz_initial.shape[0]} particles:")

#     os.makedirs("tests/tmp", exist_ok=True)
#     log_file = f"tests/tmp/N{xyz_initial.shape[0]}_minimization_with_nl.log"

#     # Build energy function with neighbor list
#     displacement_fn, _ = space.periodic(box_size)
#     neighbor_fn, energy_fn = build_energy_fn_aziz_1995_neighborlist(displacement_fn, box_size)

#     # Start timing the minimization
#     start_time = time.time()

#     results = find_local_minimum(
#         energy_fn=energy_fn,
#         xyz_initial=xyz_initial,
#         neighbor_fn=neighbor_fn,
#         log_file=log_file,
#         log_every=10
#     )

#     print(f"Initial energy: {results['energy_initial']:.8f}")

#     # End timing and calculate duration
#     end_time = time.time()
#     minimization_time = end_time - start_time

#     print(f"Iterations: {results['nit']}, Function evaluations: {results['nfev']}")
#     print(f"Minimization time: {minimization_time:.3f} seconds")
#     print(f"Performance: {results['nfev']/minimization_time:.1f} function evaluations per second")

#     assert results['success'], f"Optimization failed: {results['message']}"

#     energy_tolerance = 5e-4  # Relaxed tolerance for different optimization paths
#     energy_diff = abs(results['energy_final'] - energy_reference)
#     print(f"Final energy: {results['energy_final']:.8f}")
#     print(f"Reference energy: {energy_reference:.8f}")

#     xyz_final = results['xyz_final']
#     max_diff = jnp.max(jnp.abs(xyz_final - xyz_reference))
#     xyz_tolerance = 3.0  # Relaxed tolerance as different minima can have similar energies
#     print(f"xyz max component difference: {max_diff:.6f}")

#     assert results['energy_final'] < results['energy_initial'], \
#         "Final energy should be lower than initial energy"

#     assert energy_diff < energy_tolerance, \
#         f"Compared to the reference energy, the absolute difference {energy_diff:.2e} exceeds tolerance {energy_tolerance:.2e}"

#     assert max_diff < xyz_tolerance, \
#         f"Compared to reference xyz, the max component difference {max_diff:.6f} exceeds tolerance {xyz_tolerance}"

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
    log_file = f"tests/tmp/pimc_M{M}_N{N}.min.log"
    trajectory_file = f"tests/tmp/pimc_M{M}_N{N}.min.wl.dat"

    # Run minimization
    start_time = time.time()

    results = find_local_minimum(
        energy_fn=minimization_energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=1,
        trajectory_file=trajectory_file,
        trajectory_path_template=path_template,
        save_trajectory_every=1,
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

