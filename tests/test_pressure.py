"""
Tests for jax_landscape.pressure module.
"""

import math
import tempfile
import os

import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from jax_landscape.pressure import (
    compute_pressure_config,
    parse_wl_configs,
    HBAR2_OVER_2M_HE4,
    K_A3_TO_BAR,
)


# ── Parser tests ────────────────────────────────────────────────────────

def _write_wl_file(tmpdir, configs):
    """Write a minimal worldline file with given configs."""
    path = os.path.join(tmpdir, "test_wl.dat")
    with open(path, 'w') as f:
        for config in configs:
            f.write("# START_CONFIG\n")
            for line in config:
                f.write(line + "\n")
    return path


def test_parser_diagonal_config():
    """Parser yields a diagonal config with correct shape."""
    N, M = 2, 2
    # Format: slice ptcl wlIdx x y z linkF linkB nextSlice nextPtcl
    config = [
        "0 0 0 1.0 2.0 3.0 0 0 1 0",
        "0 1 1 4.0 5.0 6.0 0 0 1 1",
        "1 0 0 1.1 2.1 3.1 0 0 0 0",
        "1 1 1 4.1 5.1 6.1 0 0 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_wl_file(tmpdir, [config])
        results = list(parse_wl_configs(path, N, M))
        assert len(results) == 1
        pos, ns, np_arr = results[0]
        assert pos.shape == (2, 2, 3)
        assert np.isclose(pos[0, 0, 0], 1.0)
        assert np.isclose(pos[1, 1, 2], 6.1)


def test_parser_skips_offdiagonal():
    """Config with wrong bead count (N+1 particles) returns None."""
    N, M = 2, 2
    # 5 lines instead of 4 — off-diagonal
    config_offdiag = [
        "0 0 0 1.0 2.0 3.0 0 0 1 0",
        "0 1 1 4.0 5.0 6.0 0 0 1 1",
        "0 2 2 7.0 8.0 9.0 0 0 1 2",  # extra particle
        "1 0 0 1.1 2.1 3.1 0 0 0 0",
        "1 1 1 4.1 5.1 6.1 0 0 0 1",
    ]
    config_diag = [
        "0 0 0 1.0 2.0 3.0 0 0 1 0",
        "0 1 1 4.0 5.0 6.0 0 0 1 1",
        "1 0 0 1.1 2.1 3.1 0 0 0 0",
        "1 1 1 4.1 5.1 6.1 0 0 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_wl_file(tmpdir, [config_offdiag, config_diag])
        results = list(parse_wl_configs(path, N, M))
        assert len(results) == 1  # only the diagonal one


def test_parser_handles_comments():
    """Lines starting with # inside a config are skipped."""
    N, M = 2, 1
    config = [
        "# This is a comment",
        "0 0 0 1.0 2.0 3.0 0 0 0 0",
        "# Another comment",
        "0 1 1 4.0 5.0 6.0 0 0 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_wl_file(tmpdir, [config])
        results = list(parse_wl_configs(path, N, M))
        assert len(results) == 1


def test_parser_skip_configs():
    """skip_configs parameter skips the first N diagonal configs."""
    N, M = 2, 1
    config = [
        "0 0 0 1.0 2.0 3.0 0 0 0 0",
        "0 1 1 4.0 5.0 6.0 0 0 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_wl_file(tmpdir, [config, config, config])
        results = list(parse_wl_configs(path, N, M, skip_configs=2))
        assert len(results) == 1


# ── Pressure estimator tests ────────────────────────────────────────────

def test_ideal_gas():
    """With dvdr=0 (no interactions), P = NkT/V = N/(beta*V)."""
    N, M = 10, 4
    tau = 0.01  # beta = M * tau = 0.04
    beta = M * tau
    side = np.array([10.0, 10.0, 10.0])
    V = 1000.0
    lam = HBAR2_OVER_2M_HE4

    # Place particles on a grid, each bead at the same position (no springs)
    positions = np.zeros((M, N, 3))
    for i in range(N):
        positions[:, i, :] = [i * 1.0, 0.0, 0.0]

    # Connectivity: each bead links to the next slice, same particle
    next_slice = np.zeros((M, N), dtype=np.int32)
    next_ptcl = np.zeros((M, N), dtype=np.int32)
    for k in range(M):
        next_slice[k, :] = (k + 1) % M
        next_ptcl[k, :] = np.arange(N)

    # Zero force
    dvdr_fn = lambda r: np.zeros_like(r)

    P_id, P_sp, P_vir = compute_pressure_config(
        positions, next_slice, next_ptcl, N, M, side, tau, lam, 5.0, dvdr_fn)

    # With identical bead positions, spring term is zero
    assert abs(P_sp) < 1e-14
    assert abs(P_vir) < 1e-14

    # P_ideal = (1/(3*tau*M*V)) * 3*N*M = N / (tau * V)
    # Note: this is the primitive estimator formula, not the thermodynamic P=NkT/V
    P_expected = N / (tau * V)
    assert P_id == pytest.approx(P_expected, rel=1e-12)


def test_mock_2particle_2slice():
    """Hand-verified 2-particle, 2-slice pressure calculation."""
    N, M = 2, 2
    tau = 0.1
    side = np.array([10.0, 10.0, 10.0])
    V = 1000.0
    lam = HBAR2_OVER_2M_HE4

    # Two particles at fixed positions, each slice identical
    positions = np.zeros((M, N, 3))
    positions[:, 0, :] = [0.0, 0.0, 0.0]
    positions[:, 1, :] = [3.0, 0.0, 0.0]

    next_slice = np.zeros((M, N), dtype=np.int32)
    next_ptcl = np.zeros((M, N), dtype=np.int32)
    for k in range(M):
        next_slice[k, :] = (k + 1) % M
        next_ptcl[k, :] = np.arange(N)

    # Constant dvdr for simplicity
    dvdr_val = 5.0  # K/A
    dvdr_fn = lambda r: np.full_like(r, dvdr_val)

    P_id, P_sp, P_vir = compute_pressure_config(
        positions, next_slice, next_ptcl, N, M, side, tau, lam, 5.0, dvdr_fn)

    # Hand calculation:
    # P_ideal = 3*N*M / (3*tau*M*V) = N / (tau*V) -- wait
    # P_ideal = (1/(3*tau*M*V)) * 3*N*M = N/(tau*V)
    # But beta = M*tau, so N/(tau*V) = N*M/(beta*V) -- not standard
    # Actually: prefactor = 1/(3*tau*M*V)
    # P_ideal = prefactor * 3*N*M = N/(tau*V)
    # For N=2, tau=0.1, V=1000: P_ideal = 0.02

    # Spring: beads are at same position across slices, so spring=0
    # Virial: r_ij = 3.0 for the one pair, dvdr = 5.0
    #   virial_sum = M * r * dvdr = 2 * 3.0 * 5.0 = 30
    #   P_virial = -prefactor * tau * 30 = -(1/(3*0.1*2*1000)) * 0.1 * 30

    prefactor = 1.0 / (3.0 * tau * M * V)
    assert P_id == pytest.approx(prefactor * 3 * N * M, rel=1e-12)
    assert abs(P_sp) < 1e-14
    expected_virial = -prefactor * tau * (M * 3.0 * dvdr_val)
    assert P_vir == pytest.approx(expected_virial, rel=1e-10)
