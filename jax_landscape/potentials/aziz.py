"""
Aziz He-He pair potential — single source of truth.

Three parameterizations are available:

- 1979 (HFDHE2): Aziz et al., J. Chem. Phys. 70, 4330 (1979)
- 1987: Aziz, McCourt & Wong, Mol. Phys. 61, 1487 (1987)
- 1995 (HFD-B3-FCI1): Aziz, Janzen & Moldover, PRL 74, 1586 (1995)

All parameter values are taken from the C++ PIMC engine
(programs/pimc/src/potential.cpp lines 1588-1624).

Units:
    V(r)   : Kelvin
    dVdr   : K / Angstrom
    r      : Angstrom
    tail_V : K * A^3  (multiply by N^2/V for energy correction)
    tail_pressure : K / A^3
"""

import math
import jax
import jax.numpy as jnp


# ── Parameters ───────────────────────────────────────────────────────────

AZIZ_PARAMS = {
    1979: {
        'A': 0.5448504e6,
        'alpha': 13.353384,
        'beta': 0.0,
        'C6': 1.3732412,
        'C8': 0.4253785,
        'C10': 0.1781,
        'D': 1.241314,
        'epsilon': 10.8,      # K
        'rm': 2.9673,         # Angstrom
    },
    1987: {
        'A': 1.8443101e5,
        'alpha': 10.43329537,
        'beta': -2.27965105,
        'C6': 1.36745214,
        'C8': 0.42123807,
        'C10': 0.17473318,
        'D': 1.4826,
        'epsilon': 10.948,    # K
        'rm': 2.9673,         # Angstrom
    },
    1995: {
        'A': 1.86924404e5,
        'alpha': 10.5717543,
        'beta': -2.07758779,
        'C6': 1.35186623,
        'C8': 0.41495143,
        'C10': 0.17151143,
        'D': 1.438,
        'epsilon': 10.956,    # K
        'rm': 2.9683,         # Angstrom
    },
}


def get_params(year=1979):
    """Return a copy of the Aziz parameter dict for the given year."""
    if year not in AZIZ_PARAMS:
        raise ValueError(f"Unknown Aziz year {year}; choose from {sorted(AZIZ_PARAMS)}")
    return dict(AZIZ_PARAMS[year])


# ── Potential ────────────────────────────────────────────────────────────

def V(r, year=1979):
    """
    Bare Aziz pair potential V(r) in Kelvin.

    Scalar function, compatible with jax.jit and jax.grad.
    """
    p = AZIZ_PARAMS[year]
    A = p['A']
    alpha = p['alpha']
    beta = p['beta']
    C6 = p['C6']
    C8 = p['C8']
    C10 = p['C10']
    D = p['D']
    epsilon = p['epsilon']
    rm = p['rm']

    x = r / rm

    # Damping function
    Fx = jnp.where(x < D,
                   jnp.exp(-(D / x - 1.0)**2),
                   1.0)

    # Repulsive term
    repulsive = A * jnp.exp(-alpha * x + beta * x**2)

    # Dispersion term
    ix = 1.0 / x
    ix2 = ix * ix
    ix6 = ix2 * ix2 * ix2
    ix8 = ix6 * ix2
    ix10 = ix8 * ix2
    dispersion = (C6 * ix6 + C8 * ix8 + C10 * ix10) * Fx

    return epsilon * (repulsive - dispersion)


V_vec = jax.vmap(V, in_axes=(0, None))
"""Vectorized V(r) over an array of distances."""

dVdr = jax.grad(V)
"""Automatic derivative dV/dr (scalar)."""

dVdr_vec = jax.vmap(dVdr, in_axes=(0, None))
"""Vectorized dV/dr over an array of distances."""


# ── Tail corrections ────────────────────────────────────────────────────

def tail_V(rc, year=1979):
    """
    Energy tail correction in K*A^3.

    Assumes g(r) = 1 beyond rc, integrating the dispersion terms analytically.
    Matches the C++ engine (potential.cpp:1640-1645).

    Usage: E_tail = (N^2 / V) * tail_V(rc)
    """
    p = AZIZ_PARAMS[year]
    C6 = p['C6']
    C8 = p['C8']
    C10 = p['C10']
    epsilon = p['epsilon']
    rm = p['rm']

    rmorc = rm / rc
    t2 = C6 * rmorc**3 / 3.0
    t3 = C8 * rmorc**5 / 5.0
    t4 = C10 * rmorc**7 / 7.0

    return 2.0 * math.pi * epsilon * (-rm**3 * (t2 + t3 + t4))


def tail_pressure(rho, rc, year=1979):
    """
    Pressure tail correction in K/A^3.

    Uses the virial route: P_tail = (2/3)*pi*rho^2 * [rc^3 * V(rc) + 3*int_rc^inf r^2*V(r)dr].
    The integral of the dispersion part is analytic; we evaluate V(rc) numerically.
    """
    p = AZIZ_PARAMS[year]
    C6 = p['C6']
    C8 = p['C8']
    C10 = p['C10']
    epsilon = p['epsilon']
    rm = p['rm']

    xc = rc / rm
    # int_{rc}^{inf} r^2 * V_disp(r) dr  (dispersion only, g(r)=1)
    int_r2v = epsilon * rm**3 * (
        -C6 * xc**(-3) / 3.0
        - C8 * xc**(-5) / 5.0
        - C10 * xc**(-7) / 7.0
    )

    # V(rc) — use float for the potential evaluation
    v_rc = float(V(jnp.float64(rc), year=year))

    return (2.0 / 3.0) * math.pi * rho**2 * (rc**3 * v_rc + 3.0 * int_r2v)
