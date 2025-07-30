import pytest
import jax.numpy as jnp
from jax_md import space
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

from md_differentials.aziz1995 import total_energy_aziz_1995_neighbor_list, total_energy_aziz_1995_no_nl


def load_test_data(filename):
    """Loads test data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return {
        'R': jnp.array(data['xyz']),
        'box': jnp.array(data['box']),
        'energy': data['Etot']
    }


@pytest.fixture
def n6_test_data():
    """Load N6 test data."""
    return load_test_data('tests/test_data/aziz1995-N6-Nbeads1.json')


@pytest.fixture
def n500_test_data():
    """Load N500 test data."""
    return load_test_data('tests/test_data/aziz1995-N500-Nbeads1.json')


class TestAziz1995Energy:
    """Test class for Aziz 1995 potential energy calculations."""
    
    def test_print_energy_values_n6(self, n6_test_data):
        """Print energy values to verify they match reference data for N6."""
        R = n6_test_data['R']
        box_size = n6_test_data['box']
        expected_energy = n6_test_data['energy']
        
        displacement, shift = space.periodic(box_size)
        
        # Energy without neighbor list
        energy_fn_no_nl = total_energy_aziz_1995_no_nl(displacement)
        energy_no_nl = energy_fn_no_nl(R)
        
        # Energy with neighbor list
        neighbor_fn, energy_fn_nl = total_energy_aziz_1995_neighbor_list(displacement, box_size)
        nbrs = neighbor_fn.allocate(R)
        energy_nl = energy_fn_nl(R, neighbor=nbrs)
        
        print(f"\nN6 System Energy Comparison:")
        print(f"Reference energy: {expected_energy}")
        print(f"Energy (no neighbor list): {energy_no_nl}")
        print(f"Energy (with neighbor list): {energy_nl}")
        print(f"Difference from reference (no NL): {abs(energy_no_nl - expected_energy)}")
        print(f"Difference from reference (with NL): {abs(energy_nl - expected_energy)}")
        
        # Verify they match reference data
        assert jnp.isclose(energy_no_nl, expected_energy, rtol=1e-10)
        assert jnp.isclose(energy_nl, expected_energy, rtol=1e-10)
    
    def test_print_energy_values_n500(self, n500_test_data):
        """Print energy values to verify they match reference data for N500."""
        R = n500_test_data['R']
        box_size = n500_test_data['box']
        expected_energy = n500_test_data['energy']
        
        displacement, shift = space.periodic(box_size)
        
        # Energy without neighbor list
        energy_fn_no_nl = total_energy_aziz_1995_no_nl(displacement)
        energy_no_nl = energy_fn_no_nl(R)
        
        # Energy with neighbor list
        neighbor_fn, energy_fn_nl = total_energy_aziz_1995_neighbor_list(displacement, box_size)
        nbrs = neighbor_fn.allocate(R)
        energy_nl = energy_fn_nl(R, neighbor=nbrs)
        
        print(f"\nN500 System Energy Comparison:")
        print(f"Reference energy: {expected_energy}")
        print(f"Energy (no neighbor list): {energy_no_nl}")
        print(f"Energy (with neighbor list): {energy_nl}")
        print(f"Difference from reference (no NL): {abs(energy_no_nl - expected_energy)}")
        print(f"Difference from reference (with NL): {abs(energy_nl - expected_energy)}")
        
        # Verify they match reference data
        assert jnp.isclose(energy_no_nl, expected_energy, rtol=1e-10)
        assert jnp.isclose(energy_nl, expected_energy, rtol=1e-10)