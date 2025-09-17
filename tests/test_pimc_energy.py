import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import space

from jax_landscape.io.pimc import Path, load_pimc_worldline_file
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_no_neighborlist
from jax_landscape.pimc_energy import build_pimc_energy_fn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")


def test_classical_M1_matches_slice_energy():
    # Build a single-slice configuration
    N = 3
    M = 1
    wldata = jnp.array([[[0, 0, 1, 0.3, 2.4, -3.4, 0, 0, 0, 0],
                         [0, 1, 1, 3.2, -3.4, 48.123, 0, 1, 0, 1],
                         [0, 2, 1, 7.3, -9.44, -31.4, 0, 2, 0, 2]]])  #
    path = Path(wldata[0])

    beta = 1.0
    mass = 1.0
    hbar= 1.0
    box = jnp.array([5.0,5.0,5.0])
    displacement_fn, _ = space.periodic(box)
    potential_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    pimc_fn = build_pimc_energy_fn(displacement_fn, potential_fn)

    res = pimc_fn(path, beta=beta,hbar=hbar, mass=mass)
    # Potential energy should equal direct evaluation on the single slice
    V_direct = potential_fn(path.beadCoord[0]) 
    assert jnp.isclose(res['E_int'], V_direct)
    # Spring energy for M=1 path with self-links is zero (no displacement)
    assert jnp.isclose(res['E_sp'], 0.0)


def test_free_particle_one_cycle():
    wls = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle1.dat')
    path = Path(wls[0], Lx=100.0, Ly=100.0, Lz=100.0) 
    M = path.numTimeSlices

    beta = 0.5
    mass = 1.0
    hbar = 0.39183

    # Compute using pimc_energy()
    box = jnp.array([100.0,100.0,100.0])  # large to avoid wrapping
    displacement_fn, _ = space.periodic(box)

    def zero_potential(R):
        return jnp.array(0.0)

    pimc_fn = build_pimc_energy_fn(displacement_fn, zero_potential)
    res = pimc_fn(path, beta=beta, hbar=hbar, mass=mass)

    # Expected value
    sum_dr2_ref = 1 + 1 + 2 + 1 + 1 + 2  # directly counting the |dr|^2 for each link.
    expected_E_sp = 0.5 * mass * M / (beta * hbar)**2 * sum_dr2_ref

    assert jnp.isclose(res['E_sp'], expected_E_sp, rtol=1e-8)
    assert res['E_int'] == 0.0

def test_free_particle_two_cycle():
    wls = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle2.dat')
    path = Path(wls[0], Lx=100.0, Ly=100.0, Lz=100.0) 
    M = path.numTimeSlices

    beta = 0.5
    mass = 1.0
    hbar = 0.39183

    # Compute using pimc_energy()
    box = jnp.array([100.0,100.0,100.0])  # large to avoid wrapping
    displacement_fn, _ = space.periodic(box)

    def zero_potential(R):
        return jnp.array(0.0)

    pimc_fn = build_pimc_energy_fn(displacement_fn, zero_potential)
    res = pimc_fn(path, beta=beta, hbar=hbar, mass=mass)

    # Expected value
    sum_dr2_ref = 2 * 4 + 1 * 2 # directly counting the |dr|^2 for each link.
    expected_E_sp = 0.5 * mass * M / (beta * hbar)**2 * sum_dr2_ref

    assert jnp.isclose(res['E_sp'], expected_E_sp, rtol=1e-8)
    assert res['E_int'] == 0.0