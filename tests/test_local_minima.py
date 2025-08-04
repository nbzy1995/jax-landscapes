"""
Test local minimization routines.
"""

import json
import os
import pytest
import jax
import jax.numpy as jnp

from jax_landscape.local_minima import find_local_minimum
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_neighborlist, build_energy_fn_aziz_1995_no_neighborlist

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")


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
    'tests/test_data/aziz1995-N500-Nbeads1.json'
])
def test_local_minimum_aziz1995(data_file):
    """Test local minimization Helium system against reference inherent structure."""
    # Load test data
    data = load_test_data(data_file)

    xyz_initial = data['xyz']
    box_size = data['box']
    xyz_reference = data['xyz_IS']
    energy_reference = data['E_IS']
    
    print(f"\nTesting local minimization with {xyz_initial.shape[0]} particles:")

    os.makedirs("tests/tmp", exist_ok=True)
    log_file = f"tests/tmp/N{xyz_initial.shape[0]}_minimization.log"
    
    results = find_local_minimum(
        energy_fn_factory=build_energy_fn_aziz_1995_no_neighborlist,
        xyz_initial=xyz_initial,
        box_size=box_size,
        log_file=log_file,
        log_every=1
    )

    assert results['success'], f"Optimization failed: {results['message']}"
    
    assert results['energy_final'] < results['energy_initial'], \
        "Final energy should be lower than initial energy"
    
    energy_tolerance = 1e-4
    energy_diff = abs(results['energy_final'] - energy_reference)
    assert energy_diff < energy_tolerance, \
        f"Compared to the reference energy, difference {energy_diff:.2e} exceeds tolerance {energy_tolerance:.2e}"
    
    xyz_final = results['xyz_final']
    max_diff = jnp.max(jnp.abs(xyz_final - xyz_reference))
    xyz_tolerance = 0.1
    assert max_diff < xyz_tolerance, \
        f"Compared to reference xyz, max component difference {max_diff:.6f} exceeds tolerance {xyz_tolerance}"

    print(f"Initial energy: {results['energy_initial']:.8f}")
    print(f"Final energy: {results['energy_final']:.8f}")
    print(f"Reference energy: {energy_reference:.8f}")
    print(f"xyz max component difference: {max_diff:.6f}")
    print(f"Iterations: {results['nit']}, Function evaluations: {results['nfev']}")

