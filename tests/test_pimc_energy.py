import pytest
import numpy as np
import jax
import jax.numpy as jnp
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
                         [0, 2, 1, 7.3, -9.44, -31.4, 0, 2, 0, 2]]])
    path = Path(wldata[0])

    beta = 1.0
    mass = 1.0
    hbar= 1.0
    box = jnp.array([5.0,5.0,5.0])
    displacement_fn, _ = space.periodic(box)
    # Use non-neighborlist version for simplicity in single-slice test
    potential_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    pimc_fn = build_pimc_energy_fn(displacement_fn, potential_fn)

    res = pimc_fn(path, beta=beta,hbar=hbar, mass=mass)
    # Potential energy should equal direct evaluation on the single slice
    V_direct = potential_fn(path.beadCoord[0]) 
    assert jnp.isclose(res['E_int'], V_direct)
    # Spring energy for M=1 path with self-links is zero (no displacement)
    assert jnp.isclose(res['E_sp'], 0.0)


def test_free_particle_one_cycle():
    paths_dict = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle1.dat', Lx=100.0, Ly=100.0, Lz=100.0)
    path = paths_dict[0]
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
    paths_dict = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle2.dat', Lx=100.0, Ly=100.0, Lz=100.0)
    path = paths_dict[0] 
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


def test_aziz_to_N64_adrian():
    wlfile = 'tests/test_data/T1.55-N64-n0.0218.dat'

    # All quantity expressed in reduced units relative to the Helium units:
    #  length [L] in Angstrom
    #  energy [E] in kB K
    #  mass [m] in Helium mass kg
    # unless otherwise specified.
    
    # System parameters for this particular wl file

    N = 64     #
    n = 0.0218 # Density in Angstrom^-3
    T = 1.55   # temperature [K]

    beta = 1/T # reduced units.
    hbar = 21.8735/(2*np.pi)
    mass = 1

    L = (N/n)**(1/3)  # box length in Angstrom
    paths_dict = load_pimc_worldline_file(wlfile, Lx=L, Ly=L, Lz=L)
    path = paths_dict[0]  # coordinates are in Angstrom
    M = path.numTimeSlices

    # Retrieve estimator info computed from Adrian pimc code
    #  K  V   V_ext    V_int    E     E_mu    K/N   V/N   E/N
    #  7.92300269E+02 -1.41078288E+03  0.00000000E+00 -1.40492804E+03 -6.18482609E+02 -6.18482609E+02  1.23796917E+01 -2.20434825E+01 -9.66379077E+00

    ref_Eqm = -6.18482609E+02 # already in kB K
    ref_Eint = -1.40492804E+03 # already in kB K

    # Compute using pimc_energy()
    box = jnp.array([L,L,L])
    displacement_fn, _ = space.periodic(box)
    potential_fn = build_energy_fn_aziz_1995_no_neighborlist(
        displacement_fn,
        box_size=box,
        r_cutoff=7.0
    )
    pimc_fn = build_pimc_energy_fn(displacement_fn, potential_fn)
    res = pimc_fn(path, beta=beta, hbar=hbar, mass=mass)

    # print("the spring constant (m M / (beta hbar)^2) is: ", mass * M / (beta * hbar)**2)
    
    print("E_qm from pimc_energy(): ", res['E_qm'])
    print("E_qm from reference: ", ref_Eqm)
    
    print("E_int from pimc_energy(): ", res['E_int'])
    print("E_int from reference: ", ref_Eint)

    print("E_sp from pimc_energy(): ", res['E_sp'])
    print("E_sp from reference: ", -1 * (ref_Eqm - ref_Eint - 1.5 * M * N / beta))

    print("NOTE: fix this later! ")

    assert jnp.isclose(res['E_qm'], ref_Eqm, rtol=1e-6)
    assert jnp.isclose(res['E_int'], ref_Eint, rtol=1e-6)