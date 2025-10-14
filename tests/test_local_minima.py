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


def _validate_local_minimum_properties(
    energy_fn,
    xyz_final,
    results,
    system_name="System",
    grad_tolerance=1e-5,
    negative_eigenvalue_threshold=-1e-6,
    near_zero_threshold=1e-6,
    expected_num_symmetry_modes=None,
    perturbation_magnitudes=(0.01, 0.05, 0.1),
    n_perturbations_per_magnitude=20,
    stability_success_threshold=0.99
):
    """
    Validate that a configuration satisfies all 4 local minimum properties.

    Args:
        energy_fn: Energy function (scalar or dict-returning)
        xyz_final: Final minimized coordinates
        results: Results dict from find_local_minimum
        system_name: Name for display (e.g., "Classical", "PIMC")
        grad_tolerance: Tolerance for gradient norm
        negative_eigenvalue_threshold: Threshold for truly negative eigenvalues (λ < threshold is a saddle point)
        near_zero_threshold: Threshold for identifying near-zero eigenvalues from symmetry
        expected_num_symmetry_modes: Expected number of symmetry modes (near-zero eigenvalues)
        perturbation_magnitudes: Tuple of perturbation sizes to test
        n_perturbations_per_magnitude: Number of random perturbations per magnitude
        stability_success_threshold: Fraction of perturbations that must increase energy
    """
    print(f"\n{'='*70}")
    print(f"Property-Based Test: {system_name}")
    print(f"{'='*70}")

    # Auto-detect if energy function returns dict or scalar
    test_result = energy_fn(xyz_final)
    is_dict_energy = isinstance(test_result, dict)

    # ========================================================================
    # Property 1: Zero Gradient Condition
    # ========================================================================
    print(f"\n--- Property 1: Zero Gradient ---")

    if is_dict_energy:
        grad_fn = jax.grad(lambda xyz: energy_fn(xyz)['energy'])
    else:
        grad_fn = jax.grad(energy_fn)

    gradient = grad_fn(xyz_final)
    grad_norm = jnp.linalg.norm(gradient)

    print(f"  Gradient norm: {grad_norm:.6e}")

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

    key = jax.random.PRNGKey(42)
    energy_final_value = results['energy_final']
    stability_tolerance = 1e-6

    total_tests = 0
    total_passed = 0

    for magnitude in perturbation_magnitudes:
        passed = 0
        for _ in range(n_perturbations_per_magnitude):
            key, subkey = jax.random.split(key)

            perturbation = jax.random.normal(subkey, shape=xyz_final.shape) * magnitude
            xyz_perturbed = xyz_final + perturbation

            energy_result = energy_fn(xyz_perturbed)
            if is_dict_energy:
                energy_perturbed = energy_result['energy']
            else:
                energy_perturbed = energy_result

            if energy_perturbed >= energy_final_value - stability_tolerance:
                passed += 1

            total_tests += 1
            total_passed += (energy_perturbed >= energy_final_value - stability_tolerance)

        print(f"  Magnitude {magnitude:.3f} Å: {passed}/{n_perturbations_per_magnitude} perturbations have E ≥ E_final")

    success_rate = total_passed / total_tests
    print(f"  Overall: {total_passed}/{total_tests} tests passed ({success_rate*100:.1f}%)")

    assert success_rate >= stability_success_threshold, \
        f"Only {success_rate*100:.1f}% of perturbations increased energy (expected ≥{stability_success_threshold*100:.0f}%)"
    print(f"  ✓ Local stability verified (≥{stability_success_threshold*100:.0f}% perturbations increase energy)")

    # ========================================================================
    # Property 4: Hessian Positive Semi-Definite (Refined Criterion)
    # ========================================================================
    print(f"\n--- Property 4: Hessian Positive Semi-Definite ---")

    hessian_result = compute_hessian_eigenvalues(
        energy_fn,
        xyz_final,
        return_hessian=False,
        return_eigenvectors=False
    )

    eigenvalues = jnp.sort(hessian_result['eigenvalues'])

    # sufficient condition for local minimum:
    # 1. No significantly negative eigenvalues (λ < -1e-6)
    # 2. Allow near-zero eigenvalues from symmetry modes (|λ| < 1e-6)

    n_negative = jnp.sum(eigenvalues < negative_eigenvalue_threshold)
    n_near_zero = jnp.sum(jnp.abs(eigenvalues) < near_zero_threshold)
    min_eigenvalue = jnp.min(eigenvalues)
    max_eigenvalue = jnp.max(eigenvalues)

    print(f"  Total eigenvalues: {len(eigenvalues)}")
    print(f"  Near-zero eigenvalues (|λ| < {near_zero_threshold:.0e}): {n_near_zero}")
    if expected_num_symmetry_modes is not None:
        print(f"    Expected symmetry modes: {expected_num_symmetry_modes}")
    print(f"  Negative eigenvalues (λ < {negative_eigenvalue_threshold:.0e}): {n_negative}")
    print(f"  Min eigenvalue: {min_eigenvalue:.6e}")
    print(f"  Max eigenvalue: {max_eigenvalue:.6e}")

    if n_near_zero > 0:
        print(f"  Near-zero modes: {eigenvalues[:n_near_zero]}")

    # Check: no significantly negative eigenvalues
    assert n_negative == 0, \
        f"Found {n_negative} negative eigenvalue(s) below {negative_eigenvalue_threshold:.0e}. " \
        f"Min eigenvalue: {min_eigenvalue:.6e}. This indicates a saddle point, not a local minimum."

    print(f"  ✓ No negative eigenvalues < {negative_eigenvalue_threshold:.0e}")
    print(f"  ✓ Hessian is positive semi-definite (local minimum confirmed)")

    print(f"\n{'='*70}")
    print(f"All 4 properties verified ✓")
    print(f"{'='*70}")


# ============================================================================
# Comprehensive Local Minimum Tests
# ============================================================================

@pytest.mark.parametrize("data_file", [
    'tests/test_data/aziz1995-N6-Nbeads1.json',
])
def test_local_minimum_classical_no_neighborlist(data_file):
    """
    Comprehensive test of local minimization for classical Helium system.

    Tests:
    - 4 mathematical properties (zero gradient, energy decrease, local stability, Hessian PSD)
    - Comparison against reference data (xyz_IS, E_IS)
    """
    # Load test data
    data = load_test_data(data_file)
    xyz_initial = data['xyz'] * nm_to_Angstrom
    box_size = data['box'] * nm_to_Angstrom
    xyz_reference = data['xyz_IS'] * nm_to_Angstrom
    energy_reference = data['E_IS'] * KJmol_to_KBK

    print(f"\nClassical System (No Neighbor List): N = {xyz_initial.shape[0]} particles")

    # Build energy function
    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # Run minimization
    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/classical_N{xyz_initial.shape[0]}.log"

    start_time = time.time()

    results = find_local_minimum(
        energy_fn=energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=10,
        gtol=1e-6,
        energy_change_tol=1e-8
    )

    end_time = time.time()
    minimization_time = end_time - start_time

    print(f"Minimization: {results['nit']} iterations, {results['nfev']} function evals, {minimization_time:.3f}s")
    print(f"Energy: {results['energy_initial']:.8f} → {results['energy_final']:.8f}")

    assert results['success'], f"Optimization failed: {results['message']}"

    # Part 1: Validate mathematical properties (no reference data)
    _validate_local_minimum_properties(
        energy_fn=energy_fn,
        xyz_final=results['xyz_final'],
        results=results,
        system_name="Classical System (No Neighbor List)"
    )

    # Part 2: Compare against reference data
    print(f"\n{'='*70}")
    print(f"Reference Data Comparison")
    print(f"{'='*70}")

    energy_diff = abs(results['energy_final'] - energy_reference)
    xyz_diff = jnp.max(jnp.abs(results['xyz_final'] - xyz_reference))

    print(f"  Reference energy: {energy_reference:.8f}")
    print(f"  Energy difference: {energy_diff:.6e}")
    print(f"  Max xyz component difference: {xyz_diff:.6f} Å")

    # Relaxed tolerances since different optimization paths can reach different local minima
    energy_tolerance = 5e-4
    xyz_tolerance = 3.0

    assert energy_diff < energy_tolerance, \
        f"Energy difference {energy_diff:.2e} exceeds tolerance {energy_tolerance:.2e}"
    assert xyz_diff < xyz_tolerance, \
        f"Max xyz difference {xyz_diff:.6f} exceeds tolerance {xyz_tolerance}"

    print(f"  ✓ Reference comparison passed")
    print(f"{'='*70}")


@pytest.mark.parametrize("wl_file", [
    'tests/test_data/N2-Nbeads3-cycle1.dat',
    'tests/test_data/N2-Nbeads3-cycle2.dat',
])
def test_local_minimum_pimc(wl_file):
    """
    Comprehensive test of PIMC minimization.

    Tests:
    - 4 mathematical properties (zero gradient, energy decrease, local stability, Hessian PSD)
    - Trajectory file saving and loading
    """
    # Load PIMC data
    box_size_angstrom = 10.0
    paths_dict = load_pimc_worldline_file(wl_file, Lx=box_size_angstrom, Ly=box_size_angstrom, Lz=box_size_angstrom)
    path = paths_dict[0]

    M, N, _ = path.beadCoord.shape

    print(f"\nPIMC System: M={M} time slices, N={N} particles, Total beads={M*N}")

    # Setup energy functions
    displacement_fn, _ = space.periodic(box_size_angstrom)
    classical_energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # PIMC parameters (He-4 at moderate temperature)
    beta = 10.0
    hbar = 7.638
    mass = 4.0026

    # Build PIMC energy and wrap for minimization
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
    minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
        pimc_energy_fn, path, beta, hbar, mass
    )

    xyz_initial = path.beadCoord

    # Setup files
    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/pimc_M{M}_N{N}.log"
    trajectory_file = f"tests/tmp/pimc_M{M}_N{N}.wl.dat"

    # Run minimization with trajectory saving
    start_time = time.time()

    results = find_local_minimum(
        energy_fn=minimization_energy_fn,
        xyz_initial=xyz_initial,
        log_file=log_file,
        log_every=1,
        trajectory_file=trajectory_file,
        trajectory_path_template=path_template,
        save_trajectory_every=10,
        gtol=1e-6,
        energy_change_tol=1e-8,
        maxiter=10000
    )

    end_time = time.time()
    minimization_time = end_time - start_time

    print(f"Minimization: {results['nit']} iterations, {results['nfev']} function evals, {minimization_time:.3f}s")
    print(f"Energy (U_RP): {results['energy_initial']:.8f} → {results['energy_final']:.8f}")

    assert results['success'], f"Optimization failed: {results['message']}"
    assert results['xyz_final'].shape == xyz_initial.shape, "Output shape should match input"

    # Part 1: Validate mathematical properties
    # For N=2 particles, expect 5 symmetry modes (near-zero eigenvalues)
    _validate_local_minimum_properties(
        energy_fn=minimization_energy_fn,
        xyz_final=results['xyz_final'],
        results=results,
        system_name="PIMC System",
        expected_num_symmetry_modes=5 if N == 2 else None
    )

    # Part 2: Verify trajectory file
    print(f"\n{'='*70}")
    print(f"Trajectory File Verification")
    print(f"{'='*70}")

    trajectory_paths = load_pimc_worldline_file(
        trajectory_file,
        Lx=box_size_angstrom,
        Ly=box_size_angstrom,
        Lz=box_size_angstrom
    )
    num_configs = len(trajectory_paths)

    print(f"  Trajectory contains {num_configs} configurations")

    assert num_configs > 0, "Trajectory file should contain at least one configuration"

    # Verify we can access a Path from the trajectory
    first_config_key = list(trajectory_paths.keys())[0]
    first_config_path = trajectory_paths[first_config_key]

    assert first_config_path.beadCoord.shape == xyz_initial.shape, \
        "Trajectory config should have same shape as initial"

    print(f"  ✓ Trajectory file verified")
    print(f"{'='*70}")

