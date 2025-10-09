import pytest
import jax
import jax.numpy as jnp
import numpy as np
import json
from jax_md import space

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")

from jax_landscape.hessian_eigenvals import compute_hessian_eigenvalues
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_no_neighborlist
from jax_landscape.pimc_energy import build_pimc_energy_fn, build_pimc_energy_fn_xyz
from jax_landscape.io.pimc import load_pimc_worldline_file


nm_to_Angstrom = 10.0
KJmol_to_KBK = 120.2722922542


def load_test_data(filename):
    """Loads test data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return {
        'R': jnp.array(data['xyz']).reshape(-1, 3),
        'box': jnp.array(data['box']),
        'energy': data['E'],
        'grad': jnp.array(data['grad_E']).reshape(-1, 3)
    }


def test_hessian_classical_small_system():
    """Test Hessian computation on a small classical system."""
    test_data = load_test_data('tests/test_data/aziz1995-N6-Nbeads1.json')
    R = test_data['R'] * nm_to_Angstrom
    box_size = test_data['box'] * nm_to_Angstrom

    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    result = compute_hessian_eigenvalues(
        energy_fn,
        R,
        return_hessian=True,
        return_eigenvectors=True
    )

    print(f"\nClassical system: N={R.shape[0]} particles")
    print(f"Energy: {result['energy']}")
    print(f"Hessian shape: {result['hessian'].shape}")
    print(f"Number of eigenvalues: {len(result['eigenvalues'])}")
    print(f"First 6 eigenvalues: {result['eigenvalues'][:6]}")

    # Check dimensions
    D = R.shape[0] * 3  # Total DOF
    assert result['hessian'].shape == (D, D)
    assert result['eigenvalues'].shape == (D,)
    assert result['eigenvectors'].shape == (D, D)

    # Verify Hessian is symmetric
    assert jnp.allclose(result['hessian'], result['hessian'].T, atol=1e-10), \
        "Hessian should be symmetric"

    # Verify eigenvalues are sorted (ascending)
    assert jnp.all(result['eigenvalues'][1:] >= result['eigenvalues'][:-1]), \
        "Eigenvalues should be sorted in ascending order"

    # Verify energy matches
    expected_energy = test_data['energy'] * KJmol_to_KBK
    assert jnp.isclose(result['energy'], expected_energy, rtol=1e-8)


def test_hessian_without_eigenvectors():
    """Test that we can skip eigenvector computation."""
    test_data = load_test_data('tests/test_data/aziz1995-N6-Nbeads1.json')
    R = test_data['R'] * nm_to_Angstrom
    box_size = test_data['box'] * nm_to_Angstrom

    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    result = compute_hessian_eigenvalues(
        energy_fn,
        R,
        return_hessian=False,
        return_eigenvectors=False
    )

    # Should have eigenvalues but not eigenvectors or hessian
    assert 'eigenvalues' in result
    assert 'eigenvectors' not in result
    assert 'hessian' not in result
    assert 'energy' in result
    assert 'xyz_shape' in result


def test_hessian_verification_simple_harmonic():
    """Test Hessian eigenvalues for a simple harmonic potential."""
    # Simple 2-particle 1D harmonic oscillator: V = 0.5 * k * (x1 - x2)^2
    k = 10.0

    def harmonic_energy(xyz):
        # xyz shape (2, 3), but we only use x-coordinate
        r = xyz[0, 0] - xyz[1, 0]
        return 0.5 * k * r**2

    # Place particles at equilibrium
    xyz = jnp.array([[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]])

    result = compute_hessian_eigenvalues(
        harmonic_energy,
        xyz,
        return_hessian=True
    )

    print(f"\nSimple harmonic test:")
    print(f"Eigenvalues: {result['eigenvalues']}")

    # For 1D harmonic oscillator with 2 particles:
    # Hessian should have eigenvalues: [0, 0, 0, 0, 0, 2k]
    # (5 zero modes: translation + unused DOF, 1 vibration mode)
    eigenvalues = result['eigenvalues']

    # Check that we have 5 near-zero eigenvalues and 1 non-zero
    near_zero = jnp.sum(jnp.abs(eigenvalues) < 1e-6)
    assert near_zero == 5, f"Expected 5 zero eigenvalues, got {near_zero}"

    # The largest eigenvalue should be 2*k
    assert jnp.isclose(eigenvalues[-1], 2*k, rtol=1e-6), \
        f"Expected largest eigenvalue {2*k}, got {eigenvalues[-1]}"


def test_hessian_pimc_energy():
    """Test Hessian computation with PIMC energy function."""
    wlfile = 'tests/test_data/N2-Nbeads3-cycle1.dat'
    paths_dict = load_pimc_worldline_file(wlfile, Lx=100.0, Ly=100.0, Lz=100.0)
    path = paths_dict[0]

    beta = 0.5
    mass = 1.0
    hbar = 0.39183
    box = jnp.array([100.0, 100.0, 100.0])

    displacement_fn, _ = space.periodic(box)

    # Use zero potential for simplicity
    def zero_potential(R):
        return jnp.array(0.0)

    pimc_fn = build_pimc_energy_fn(displacement_fn, zero_potential)
    minimization_energy_fn, _ = build_pimc_energy_fn_xyz(
        pimc_fn, path, beta, hbar, mass
    )

    result = compute_hessian_eigenvalues(
        minimization_energy_fn,
        path.beadCoord,
        return_hessian=False,
        return_eigenvectors=True
    )

    M, N = path.beadCoord.shape[0], path.beadCoord.shape[1]
    D = M * N * 3

    print(f"\nPIMC system: M={M} time slices, N={N} particles")
    print(f"Total DOF: {D}")
    print(f"Number of eigenvalues: {len(result['eigenvalues'])}")
    print(f"First 6 eigenvalues: {result['eigenvalues'][:6]}")

    assert result['eigenvalues'].shape == (D,)
    assert result['eigenvectors'].shape == (D, D)
    assert result['xyz_shape'] == path.beadCoord.shape

    # For free particle PIMC, spring energy is harmonic, so Hessian should be positive semi-definite
    # (all eigenvalues >= 0, with some near-zero for translation modes)
    assert jnp.all(result['eigenvalues'] >= -1e-8), \
        "Eigenvalues should be non-negative for quadratic spring potential"


def test_numerical_gradient_consistency():
    """Verify that Hessian via autodiff is consistent with numerical differentiation."""
    # Very small system for numerical test
    xyz = jnp.array([[0.0, 0.0, 0.0],
                     [3.0, 0.0, 0.0]])
    box_size = jnp.array([10.0, 10.0, 10.0])

    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    # Compute Hessian via autodiff
    result = compute_hessian_eigenvalues(
        energy_fn,
        xyz,
        return_hessian=True
    )
    hessian_auto = result['hessian']

    # Compute Hessian numerically (finite differences)
    eps = 1e-5
    D = xyz.size
    xyz_flat = xyz.flatten()
    hessian_numerical = jnp.zeros((D, D))

    for i in range(D):
        for j in range(D):
            # f(x + ei + ej)
            xyz_pp = xyz_flat.at[i].add(eps).at[j].add(eps)
            f_pp = energy_fn(xyz_pp.reshape(xyz.shape))

            # f(x + ei - ej)
            xyz_pm = xyz_flat.at[i].add(eps).at[j].add(-eps)
            f_pm = energy_fn(xyz_pm.reshape(xyz.shape))

            # f(x - ei + ej)
            xyz_mp = xyz_flat.at[i].add(-eps).at[j].add(eps)
            f_mp = energy_fn(xyz_mp.reshape(xyz.shape))

            # f(x - ei - ej)
            xyz_mm = xyz_flat.at[i].add(-eps).at[j].add(-eps)
            f_mm = energy_fn(xyz_mm.reshape(xyz.shape))

            # Central difference formula for mixed partial derivative
            hessian_numerical = hessian_numerical.at[i, j].set(
                (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            )

    print(f"\nNumerical Hessian consistency test:")
    print(f"Max absolute difference: {jnp.max(jnp.abs(hessian_auto - hessian_numerical))}")
    print(f"Relative difference: {jnp.max(jnp.abs(hessian_auto - hessian_numerical)) / jnp.max(jnp.abs(hessian_auto))}")

    # Hessians should match within numerical precision
    assert jnp.allclose(hessian_auto, hessian_numerical, rtol=1e-4, atol=1e-6), \
        "Autodiff Hessian should match numerical Hessian"


def test_eigenvalue_eigenvector_reconstruction():
    """Verify that H * v = lambda * v for computed eigenvalues/eigenvectors."""
    test_data = load_test_data('tests/test_data/aziz1995-N6-Nbeads1.json')
    R = test_data['R'] * nm_to_Angstrom
    box_size = test_data['box'] * nm_to_Angstrom

    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

    result = compute_hessian_eigenvalues(
        energy_fn,
        R,
        return_hessian=True,
        return_eigenvectors=True
    )

    H = result['hessian']
    eigenvalues = result['eigenvalues']
    eigenvectors = result['eigenvectors']

    # Check H * v_i = lambda_i * v_i for a few eigenvectors
    for i in [0, len(eigenvalues)//2, -1]:
        v = eigenvectors[:, i]
        lambda_i = eigenvalues[i]
        Hv = H @ v
        lambda_v = lambda_i * v

        error = jnp.linalg.norm(Hv - lambda_v)
        print(f"\nEigenvector {i}: lambda={lambda_i:.6e}, reconstruction error={error:.6e}")

        assert jnp.allclose(Hv, lambda_v, rtol=1e-6, atol=1e-8), \
            f"Eigenvector {i} should satisfy H*v = lambda*v"
