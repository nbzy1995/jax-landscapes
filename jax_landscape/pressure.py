"""
Primitive thermodynamic pressure estimator for PIMC worldline configurations.

Computes the pressure using Tuckerman (2023) Eq. 12.3.24:

  P = (1/(3*tau*M*V)) * [3*N*M
      - (1/(2*lambda*tau)) * sum_k sum_i |r_i^(k+1) - r_i^(k)|^2
      - tau * sum_k sum_{i<j} r_ij * v'(r_ij) ]

  + tail correction

**Note:** The primitive estimator has O(tau^2) bias when applied to
configurations sampled with the Gaussian short-time (GSF) propagator.

Units throughout: Kelvin, Angstrom, bar.
"""

import os
import json
import glob as globmod
import numpy as np

from .potentials.aziz import dVdr_vec as _dVdr_vec_jax, tail_pressure as _tail_pressure


# ── Physical constants ───────────────────────────────────────────────────

HBAR2_OVER_2M_HE4 = 6.0596    # lambda = hbar^2/(2*m) in A^2*K for He-4
K_A3_TO_BAR = 138.065          # 1 K/A^3 = 138.065 bar


# ── dV/dr wrapper (numpy) ───────────────────────────────────────────────

def _dvdr_numpy(r_array, year=1979):
    """Evaluate dV/dr at an array of r values, returning numpy."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    r_jax = jnp.array(r_array, dtype=jnp.float64)
    return np.asarray(_dVdr_vec_jax(r_jax, year))


# ── Worldline parser ────────────────────────────────────────────────────

def parse_wl_configs(wl_file, N, M, max_configs=None, skip_configs=0):
    """
    Streaming parser for PIMC worldline configuration files.

    Yields (positions, next_slice, next_ptcl) tuples for diagonal
    (worm-closed) configurations only. Off-diagonal configs are silently
    skipped.

    Parameters
    ----------
    wl_file : str
        Path to worldline file.
    N : int
        Expected number of particles.
    M : int
        Number of imaginary-time slices.
    max_configs : int or None
        Maximum number of diagonal configs to yield.
    skip_configs : int
        Skip this many diagonal configs before yielding (equilibration).

    Yields
    ------
    (positions, next_slice, next_ptcl) : tuple of ndarray
        positions : (M, N, 3) float64
        next_slice : (M, N) int32
        next_ptcl : (M, N) int32
    """
    yielded = 0
    config_lines = []
    in_config = False

    with open(wl_file, 'r') as f:
        for line in f:
            if '# START_CONFIG' in line:
                if in_config and config_lines:
                    result = _try_parse_config(config_lines, N, M)
                    if result is not None:
                        if skip_configs > 0:
                            skip_configs -= 1
                        else:
                            yield result
                            yielded += 1
                            if max_configs and yielded >= max_configs:
                                return
                config_lines = []
                in_config = True
                continue

            if in_config and not line.startswith('#'):
                config_lines.append(line)

        # Last config
        if in_config and config_lines:
            result = _try_parse_config(config_lines, N, M)
            if result is not None and skip_configs <= 0:
                yield result


def _try_parse_config(lines, N, M):
    """Parse one config block. Returns None if off-diagonal."""
    if len(lines) != N * M:
        return None

    positions = np.zeros((M, N, 3))
    next_slice = np.zeros((M, N), dtype=np.int32)
    next_ptcl = np.zeros((M, N), dtype=np.int32)

    for raw_line in lines:
        parts = raw_line.split()
        if len(parts) < 10:
            return None
        sl = int(parts[0])
        pt = int(parts[1])
        if pt >= N:
            return None  # extra particle — off-diagonal
        positions[sl, pt, 0] = float(parts[3])
        positions[sl, pt, 1] = float(parts[4])
        positions[sl, pt, 2] = float(parts[5])
        ns = int(parts[8])
        np_ = int(parts[9])
        if np_ >= N:
            return None
        next_slice[sl, pt] = ns
        next_ptcl[sl, pt] = np_

    return positions, next_slice, next_ptcl


# ── Core pressure estimator ─────────────────────────────────────────────

def compute_pair_virial_slice(pos_slice, side, rc2, dvdr_fn):
    """Compute sum_{i<j} r_ij * v'(r_ij) for one time slice."""
    N = pos_slice.shape[0]

    diff = pos_slice[:, np.newaxis, :] - pos_slice[np.newaxis, :, :]
    diff -= side * np.round(diff / side)
    r2_full = np.sum(diff * diff, axis=2)

    i_idx, j_idx = np.triu_indices(N, k=1)
    r2_pairs = r2_full[i_idx, j_idx]

    mask = r2_pairs < rc2
    r_pairs = np.sqrt(r2_pairs[mask])

    dvdr = dvdr_fn(r_pairs)
    return np.sum(r_pairs * dvdr)


def compute_pressure_config(positions, next_slice, next_ptcl,
                            N, M, side, tau, lam, rc, dvdr_fn):
    """
    Compute primitive thermodynamic pressure for one configuration.

    Parameters
    ----------
    positions : (M, N, 3) array
    next_slice, next_ptcl : (M, N) arrays
        Connectivity arrays from worldline file.
    N : int
        Number of particles.
    M : int
        Number of time slices.
    side : (3,) array
        Box dimensions [Lx, Ly, Lz].
    tau : float
        Imaginary time step.
    lam : float
        lambda = hbar^2/(2m) in A^2*K.
    rc : float
        Pair cutoff radius.
    dvdr_fn : callable
        Function mapping r_array -> dV/dr array.

    Returns
    -------
    (P_ideal, P_spring, P_virial) : tuple of float
        Pressure components in K/A^3.
    """
    V = side[0] * side[1] * side[2]
    rc2 = rc * rc

    # Spring term
    spring_sum = 0.0
    for k in range(M):
        pos_next = positions[next_slice[k], next_ptcl[k]]
        dr = pos_next - positions[k]
        dr -= side * np.round(dr / side)
        spring_sum += np.sum(dr * dr)

    # Pair virial (slice by slice)
    virial_sum = 0.0
    for k in range(M):
        virial_sum += compute_pair_virial_slice(positions[k], side, rc2, dvdr_fn)

    prefactor = 1.0 / (3.0 * tau * M * V)
    P_ideal = prefactor * 3.0 * N * M
    P_spring = -prefactor * (1.0 / (2.0 * lam * tau)) * spring_sum
    P_virial = -prefactor * tau * virial_sum

    return P_ideal, P_spring, P_virial


# ── High-level interface ────────────────────────────────────────────────

def compute_pressure_from_run(run_dir, max_configs=50, skip_configs=5,
                              year=1979):
    """
    Compute pressure from a PIMC run directory.

    Reads params.json and worldline files, computes per-config pressure
    estimates, and returns the mean with correlated error bars and tail
    correction.

    Parameters
    ----------
    run_dir : str
        Path to PIMC run directory (must contain params.json and OUTPUT/).
    max_configs : int
        Max diagonal configurations to process.
    skip_configs : int
        Skip initial configs (equilibration).
    year : int
        Aziz potential year.

    Returns
    -------
    dict with keys:
        P_mean_bar, P_err_bar : mean pressure and standard error (bar)
        P_tail_bar : tail correction (bar)
        P_total_bar : P_mean + P_tail (bar)
        n_configs : number of configs processed
        components : dict with P_ideal, P_spring, P_virial means (bar)
    """
    params_file = os.path.join(run_dir, "params.json")
    with open(params_file) as pf:
        params = json.load(pf)

    N = params["N"]
    M = params["M"]
    rho = params["rho"]
    tau = params["dt"]
    side = np.array([params["Lx"], params["Ly"], params["Lz"]])
    rc = params["potential_cutoff"]
    lam = HBAR2_OVER_2M_HE4

    dvdr_fn = lambda r: _dvdr_numpy(r, year=year)

    wl_files = globmod.glob(os.path.join(run_dir, "OUTPUT", "ce-wl-*.dat"))
    if not wl_files:
        raise FileNotFoundError(f"No worldline files in {run_dir}/OUTPUT/")

    P_totals = []
    P_components = []

    for pos, ns, np_arr in parse_wl_configs(
            wl_files[0], N, M,
            max_configs=max_configs, skip_configs=skip_configs):
        P_id, P_sp, P_vir = compute_pressure_config(
            pos, ns, np_arr, N, M, side, tau, lam, rc, dvdr_fn)
        P_totals.append(P_id + P_sp + P_vir)
        P_components.append((P_id, P_sp, P_vir))

    if not P_totals:
        raise ValueError(f"No diagonal configs found in {run_dir}")

    P_totals = np.array(P_totals)
    comps = np.array(P_components)
    n_cfg = len(P_totals)

    P_tail = _tail_pressure(rho, rc, year=year)

    return {
        'P_mean_bar': float(P_totals.mean() * K_A3_TO_BAR),
        'P_err_bar': float(P_totals.std() / np.sqrt(n_cfg) * K_A3_TO_BAR),
        'P_tail_bar': float(P_tail * K_A3_TO_BAR),
        'P_total_bar': float((P_totals.mean() + P_tail) * K_A3_TO_BAR),
        'n_configs': n_cfg,
        'components': {
            'P_ideal_bar': float(comps[:, 0].mean() * K_A3_TO_BAR),
            'P_spring_bar': float(comps[:, 1].mean() * K_A3_TO_BAR),
            'P_virial_bar': float(comps[:, 2].mean() * K_A3_TO_BAR),
        },
    }
