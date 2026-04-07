"""
Comprehensive tests for jax_landscape.potentials.aziz.

Tests V(r), dV/dr, and tail corrections against the C++ reference
fixture (tests/test_data/aziz_cpp_reference.json) generated from
the exact C++ formulas in programs/pimc/src/potential.cpp.
"""

import json
import math

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_landscape.potentials.aziz import (
    AZIZ_PARAMS, V, V_vec, dVdr, dVdr_vec, get_params, tail_V, tail_pressure
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cpp_reference():
    with open("tests/test_data/aziz_cpp_reference.json") as f:
        return json.load(f)


YEARS = [1979, 1987, 1995]
PARAM_KEYS = {'A', 'alpha', 'beta', 'C6', 'C8', 'C10', 'D', 'epsilon', 'rm'}


# ── Parameter tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("year", YEARS)
def test_params_all_years(year):
    """Each year has all 9 required keys."""
    p = get_params(year)
    assert set(p.keys()) == PARAM_KEYS


@pytest.mark.parametrize("year", YEARS)
def test_params_match_cpp_reference(year, cpp_reference):
    """Every parameter matches the C++ reference fixture."""
    ref_params = cpp_reference[str(year)]["params"]
    p = get_params(year)
    for key in PARAM_KEYS:
        assert p[key] == pytest.approx(ref_params[key], rel=1e-12), \
            f"year={year}, key={key}: {p[key]} != {ref_params[key]}"


def test_params_returns_copy():
    """get_params returns a copy, not the original dict."""
    p = get_params(1979)
    p['epsilon'] = 999.0
    assert AZIZ_PARAMS[1979]['epsilon'] != 999.0


def test_params_invalid_year():
    with pytest.raises(ValueError, match="Unknown Aziz year"):
        get_params(2000)


# ── V(r) correctness ────────────────────────────────────────────────────

@pytest.mark.parametrize("year", YEARS)
def test_V_against_cpp_reference(year, cpp_reference):
    """V(r) matches C++ at all grid points to machine precision."""
    ref = cpp_reference[str(year)]
    for r, v_ref in zip(ref["r_grid"], ref["V"]):
        v_jax = float(V(jnp.float64(r), year=year))
        if abs(v_ref) > 1e-15:
            assert v_jax == pytest.approx(v_ref, rel=1e-12), \
                f"year={year}, r={r}: V={v_jax} != {v_ref}"


@pytest.mark.parametrize("year", YEARS)
def test_V_repulsive_core(year):
    """V(0.5*rm) is large and positive (repulsive core)."""
    rm = AZIZ_PARAMS[year]['rm']
    v = float(V(jnp.float64(0.5 * rm), year=year))
    assert v > 0
    assert v > 100  # should be very repulsive


@pytest.mark.parametrize("year", YEARS)
def test_V_minimum_near_rm(year):
    """V(rm) is approximately -epsilon."""
    p = AZIZ_PARAMS[year]
    v = float(V(jnp.float64(p['rm']), year=year))
    assert v == pytest.approx(-p['epsilon'], rel=1e-3)


@pytest.mark.parametrize("year", YEARS)
def test_V_long_range_attractive(year):
    """V(5*rm) is small, negative (attractive tail)."""
    rm = AZIZ_PARAMS[year]['rm']
    v = float(V(jnp.float64(5.0 * rm), year=year))
    assert v < 0
    assert abs(v) < 0.01  # very weak


@pytest.mark.parametrize("year", YEARS)
def test_V_continuity_at_D(year):
    """No discontinuity at the damping boundary r = D*rm."""
    p = AZIZ_PARAMS[year]
    r_D = p['D'] * p['rm']
    eps = 1e-6
    v_below = float(V(jnp.float64(r_D - eps), year=year))
    v_above = float(V(jnp.float64(r_D + eps), year=year))
    assert abs(v_below - v_above) < 1e-3


@pytest.mark.parametrize("year", YEARS)
def test_V_vec_matches_scalar(year):
    """Vectorized V matches scalar V."""
    r_arr = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64)
    v_vec = V_vec(r_arr, year)
    for i, r in enumerate(r_arr):
        v_scalar = V(r, year)
        assert float(v_vec[i]) == pytest.approx(float(v_scalar), rel=1e-14)


# ── dV/dr correctness ───────────────────────────────────────────────────

@pytest.mark.parametrize("year", YEARS)
def test_dVdr_against_cpp_reference(year, cpp_reference):
    """dV/dr matches C++ at all grid points."""
    ref = cpp_reference[str(year)]
    for r, dv_ref in zip(ref["r_grid"], ref["dVdr"]):
        dv_jax = float(dVdr(jnp.float64(r), year=year))
        if abs(dv_ref) > 1e-8:
            assert dv_jax == pytest.approx(dv_ref, rel=1e-6), \
                f"year={year}, r={r}: dVdr={dv_jax} != {dv_ref}"


@pytest.mark.parametrize("year", YEARS)
def test_dVdr_finite_difference(year):
    """jax.grad(V) matches central finite differences."""
    rm = AZIZ_PARAMS[year]['rm']
    test_r = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]
    h = 1e-6
    for r in test_r:
        r_jax = jnp.float64(r)
        dv_auto = float(dVdr(r_jax, year=year))
        dv_fd = float((V(r_jax + h, year=year) - V(r_jax - h, year=year)) / (2 * h))
        if abs(dv_auto) > 1e-10:
            rel_err = abs(dv_auto - dv_fd) / abs(dv_auto)
            assert rel_err < 1e-7, \
                f"year={year}, r={r}: autodiff={dv_auto}, fd={dv_fd}, rel_err={rel_err}"


@pytest.mark.parametrize("year", YEARS)
def test_dVdr_zero_at_minimum(year):
    """dV/dr is approximately zero at r = rm."""
    rm = AZIZ_PARAMS[year]['rm']
    dv = float(dVdr(jnp.float64(rm), year=year))
    assert abs(dv) < 0.5  # should be very close to zero


@pytest.mark.parametrize("year", YEARS)
def test_dVdr_sign_convention(year):
    """dV/dr is negative for r < rm (repulsive slope), positive for r > rm."""
    rm = AZIZ_PARAMS[year]['rm']
    # Well inside repulsive region
    dv_inside = float(dVdr(jnp.float64(0.8 * rm), year=year))
    assert dv_inside < 0, f"Expected negative dVdr inside rm, got {dv_inside}"
    # Outside minimum
    dv_outside = float(dVdr(jnp.float64(1.2 * rm), year=year))
    assert dv_outside > 0, f"Expected positive dVdr outside rm, got {dv_outside}"


@pytest.mark.parametrize("year", YEARS)
def test_dVdr_vec_matches_scalar(year):
    """Vectorized dVdr matches scalar dVdr."""
    r_arr = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64)
    dv_vec = dVdr_vec(r_arr, year)
    for i, r in enumerate(r_arr):
        dv_scalar = dVdr(r, year)
        assert float(dv_vec[i]) == pytest.approx(float(dv_scalar), rel=1e-12)


# ── Tail corrections ────────────────────────────────────────────────────

@pytest.mark.parametrize("year", YEARS)
def test_tail_V_matches_cpp_formula(year, cpp_reference):
    """tail_V(rc=9) matches C++ tailV formula."""
    ref = cpp_reference[str(year)]
    tv = tail_V(9.0, year=year)
    assert tv == pytest.approx(ref["tailV_rc_9"], rel=1e-12)


@pytest.mark.parametrize("year", YEARS)
def test_tail_V_is_negative(year):
    """Energy tail correction should be negative (attractive tail)."""
    tv = tail_V(9.0, year=year)
    assert tv < 0


@pytest.mark.parametrize("year", YEARS)
def test_tail_pressure_vs_quadrature(year):
    """Pressure tail correction matches numerical integration."""
    from scipy import integrate

    rc = 9.0
    rho = 0.0218  # typical liquid density

    # Numerical integration of the virial route
    def integrand(r):
        return r**3 * float(dVdr(jnp.float64(r), year=year))

    integral, _ = integrate.quad(integrand, rc, 100.0, limit=200)
    P_tail_quad = -(2.0 / 3.0) * math.pi * rho**2 * integral

    P_tail_analytic = tail_pressure(rho, rc, year=year)

    # The analytic formula assumes g(r)=1, so this should match well
    assert P_tail_analytic == pytest.approx(P_tail_quad, rel=1e-3), \
        f"year={year}: analytic={P_tail_analytic}, quad={P_tail_quad}"
