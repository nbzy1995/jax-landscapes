import pytest
import jax.numpy as jnp
from jax_md import space
import json

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")

from jax_landscape.energy_fun import (
    build_energy_fn_aziz_1995_neighborlist,
    build_energy_fn_aziz_1995_no_neighborlist,
    build_energy_fn_aziz,
    aziz_1995,
)
from jax_landscape.potentials.aziz import V


nm_to_Angstrom = 10.0
KJmol_to_KBK = 120.2722922542  # kJ/mol to kB K

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


@pytest.mark.parametrize("data_file", [
    'tests/test_data/aziz1995-N6-Nbeads1.json',
    'tests/test_data/aziz1995-N500-Nbeads1.json'
])
def test_energy_grad(data_file):
    """
    Test energy and grad given coordinates xyz, with or without neighborlist, to verify they match reference data.
    """
    # test data is in units of
    # nm, kJ/mol
    test_data = load_test_data(data_file)
    R = test_data['R'] * nm_to_Angstrom
    box_size = test_data['box'] * nm_to_Angstrom
    expected_energy = test_data['energy'] * KJmol_to_KBK
    expected_grad = test_data['grad'] * (KJmol_to_KBK/nm_to_Angstrom)

    print(f"\nTesting with {R.shape[0]} particles in a box of size {box_size}:\n")
    displacement, _ = space.periodic(box_size)
    
    # Energy without neighbor list
    energy_fn_no_nl = build_energy_fn_aziz_1995_no_neighborlist(displacement)
    energy_no_nl = energy_fn_no_nl(R)
    grad_no_nl = jax.grad(energy_fn_no_nl)(R)
    
    # Energy with neighbor list
    neighbor_fn, energy_fn_nl = build_energy_fn_aziz_1995_neighborlist(displacement, box_size)
    nbrs = neighbor_fn.allocate(R)
    energy_nl = energy_fn_nl(R, neighbor=nbrs)
    grad_nl = jax.grad(energy_fn_nl, argnums=0)(R, neighbor=nbrs)

    print(f"\nTotal Energy:")
    print(f"Reference energy: {expected_energy}")
    print(f"Energy (no neighbor list)  : {energy_no_nl}")
    print(f"Energy (with neighbor list): {energy_nl}")
    print(f"Relative Difference from reference (no NL)  : {abs(energy_no_nl - expected_energy)/expected_energy}")
    print(f"Relative Difference from reference (with NL): {abs(energy_nl - expected_energy)/expected_energy}")
    print(f"Relative Difference between neighbor, without neighbor: {abs(energy_no_nl - energy_nl)/energy_no_nl}")
    assert jnp.isclose(energy_no_nl, expected_energy, rtol=1e-8)
    assert jnp.isclose(energy_nl, expected_energy, rtol=1e-8)
    assert jnp.isclose(energy_no_nl, energy_nl, rtol=1e-8), "Methods should agree with float64"

    print(f"\nGradients:")
    print(f"Relative Max components difference from reference (no neighbor list)  : {abs(grad_no_nl - expected_grad).max()/abs(expected_grad).mean()}")
    print(f"Relative Max components difference from reference (with neighbor list): {abs(grad_nl - expected_grad).max()/abs(expected_grad).mean()}")
    print(f"Relative Max components difference between neighbor , without neighbor: {abs(grad_nl - grad_no_nl).max()/abs(grad_no_nl).mean()}")
    assert jnp.allclose(grad_no_nl, expected_grad, rtol=1e-8), "Gradients should match reference data"
    assert jnp.allclose(grad_nl, expected_grad, rtol=1e-8), "Gradients should match reference data"
    assert jnp.allclose(grad_no_nl, grad_nl, rtol=1e-8), "Gradient methods should agree with float64"


# ── New tests for generic factory and backward compatibility ─────────────

def test_generic_factory_matches_1995_factory():
    """build_energy_fn_aziz(year=1995) matches the legacy 1995 factory."""
    data = load_test_data('tests/test_data/aziz1995-N6-Nbeads1.json')
    R = data['R'] * nm_to_Angstrom
    box_size = data['box'] * nm_to_Angstrom

    displacement, _ = space.periodic(box_size)

    e_legacy = build_energy_fn_aziz_1995_no_neighborlist(displacement)
    e_generic = build_energy_fn_aziz(displacement, year=1995)

    assert jnp.isclose(e_legacy(R), e_generic(R), rtol=1e-14)


def test_aziz_1979_factory():
    """build_energy_fn_aziz(year=1979) runs without error."""
    box_size = jnp.array([20.0, 20.0, 20.0])
    displacement, _ = space.periodic(box_size)

    e_fn = build_energy_fn_aziz(displacement, year=1979, r_cutoff=10.0)
    R = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
    E = e_fn(R)
    assert jnp.isfinite(E)


def test_switching_disabled():
    """With r_sw=r_cutoff, energy matches bare potential within cutoff."""
    box_size = jnp.array([20.0, 20.0, 20.0])
    displacement, _ = space.periodic(box_size)

    r_cutoff = 9.0
    e_fn = build_energy_fn_aziz(
        displacement, year=1995, r_cutoff=r_cutoff, r_sw=r_cutoff)

    # Two particles at r = 3.0 (well within cutoff)
    R = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
    E_switched = float(e_fn(R))
    E_bare = float(V(jnp.float64(3.0), year=1995))

    assert E_switched == pytest.approx(E_bare, rel=1e-10)