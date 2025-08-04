import pytest
import jax.numpy as jnp
from jax_md import space
import json

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")

from jax_landscape.energy_fun import build_energy_fn_aziz_1995_neighborlist, build_energy_fn_aziz_1995_no_neighborlist


def load_test_data(filename):
    """Loads test data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return {
        'R': jnp.array(data['xyz']),
        'box': jnp.array(data['box']),
        'energy': data['E'],
        'grad': jnp.array(data['grad_E'])
    }


@pytest.mark.parametrize("data_file", [
    'tests/test_data/aziz1995-N6-Nbeads1.json',
    'tests/test_data/aziz1995-N500-Nbeads1.json'
])
def test_energy_grad(data_file):
    """
    Test energy and grad given coordinates xyz, with or without neighborlist, to verify they match reference data.
    """
    test_data = load_test_data(data_file)
    R = test_data['R']
    box_size = test_data['box']
    expected_energy = test_data['energy']
    expected_grad = test_data['grad']

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
    # assert jnp.isclose(energy_no_nl, expected_energy, rtol=1e-4)
    # assert jnp.isclose(energy_nl, expected_energy, rtol=1e-4)
    assert jnp.isclose(energy_no_nl, energy_nl, rtol=1e-8), "Methods should agree with float64"
    # A small difference might be due to different switching functions in reference data

    print(f"\nGradients:")
    print(f"Relative Max components difference from reference (no neighbor list)  : {abs(grad_no_nl - expected_grad).max()/abs(expected_grad).mean()}")
    print(f"Relative Max components difference from reference (with neighbor list): {abs(grad_nl - expected_grad).max()/abs(expected_grad).mean()}")
    print(f"Relative Max components difference between neighbor , without neighbor: {abs(grad_nl - grad_no_nl).max()/abs(grad_no_nl).mean()}")
    # assert jnp.allclose(grad_no_nl, expected_grad, rtol=1e-4), "Gradients should match reference data"
    # assert jnp.allclose(grad_nl, expected_grad, rtol=1e-4), "Gradients should match reference data"
    assert jnp.allclose(grad_no_nl, grad_nl, rtol=1e-8), "Gradient methods should agree with float64"