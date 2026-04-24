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

def _ensure_jax_f64():
    """Enable float64 in JAX (idempotent)."""
    import jax
    jax.config.update('jax_enable_x64', True)

_ensure_jax_f64()


def _dvdr_numpy(r_array, year=1979):
    """Evaluate dV/dr at an array of r values, returning numpy."""
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

def compute_pressure_config(positions, next_slice, next_ptcl,
                            N, M, side, tau, lam, rc, dvdr_fn):
    """
    Compute primitive thermodynamic pressure for one configuration.

    Fully vectorized over time slices: pair distances from all M slices
    are collected and passed to dvdr_fn in a single call.

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

    # ── Spring term (vectorized over all slices) ──
    # Build (M, N, 3) array of next-bead positions using connectivity
    pos_next = positions[next_slice, next_ptcl]  # (M, N, 3)
    dr = pos_next - positions
    dr -= side * np.round(dr / side)
    spring_sum = np.sum(dr * dr)

    # ── Pair virial (vectorized over all slices) ──
    # positions: (M, N, 3) -> pairwise differences for all slices at once
    # diff[k, i, j, :] = pos[k,i,:] - pos[k,j,:] for upper triangle
    i_idx, j_idx = np.triu_indices(N, k=1)
    # (M, n_pairs, 3)
    diff = positions[:, i_idx, :] - positions[:, j_idx, :]
    diff -= side * np.round(diff / side)
    r2 = np.sum(diff * diff, axis=2)  # (M, n_pairs)

    mask = r2 < rc2
    r_within = np.sqrt(r2[mask])  # flat array of all within-cutoff distances

    if r_within.size > 0:
        dvdr_vals = dvdr_fn(r_within)
        virial_sum = np.sum(r_within * dvdr_vals)
    else:
        virial_sum = 0.0

    prefactor = 1.0 / (3.0 * tau * M * V)
    P_ideal = prefactor * 3.0 * N * M
    P_spring = -prefactor * (1.0 / (2.0 * lam * tau)) * spring_sum
    P_virial = -prefactor * tau * virial_sum

    return P_ideal, P_spring, P_virial


# ── Correlated error analysis ──────────────────────────────────────────

def _correlated_error(x):
    """Compute correlated standard error using integrated autocorrelation time.

    Uses Sokal's automatic windowing (positive initial sequence).

    Returns (tau_int, sigma_corr).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 1.0, float(x.std(ddof=1) / np.sqrt(n))

    y = x - x.mean()
    c0 = np.dot(y, y) / n
    if c0 == 0:
        return 1.0, 0.0

    # ACF via FFT
    fft_y = np.fft.rfft(y, n=2 * n)
    acf_full = np.fft.irfft(fft_y * np.conj(fft_y))[:n] / (c0 * n)

    # Integrated autocorrelation time (positive initial sequence)
    tau_int = 0.5
    for lag in range(1, n):
        if acf_full[lag] <= 0:
            break
        tau_int += acf_full[lag]

    naive_err = np.std(x, ddof=1) / np.sqrt(n)
    corr_err = naive_err * np.sqrt(2 * tau_int)

    return float(tau_int), float(corr_err)


def _blocking_error(x, min_blocks=4, plateau_tol=0.05, plateau_window=3):
    """Flyvbjerg-Petersen blocking analysis for a correlated time series.

    Pairwise block-averages the series until < ``min_blocks`` blocks remain,
    recording the standard error on the mean at each block size. Returns
    the error the caller should report and the full curve for diagnostics.

    Reporting rule:
      * If the last ``plateau_window`` SEMs agree within ``plateau_tol``
        (relative), report their mean as the plateau error.
      * Otherwise, report the largest observed block SEM as a conservative
        lower bound (the curve is still rising — the true SEM is at least
        this big).
    In both cases, ``plateau_reached`` tells the caller which branch fired.

    Returns
    -------
    dict with keys
      'err'              : reported SEM (float)
      'err_lastblock'    : SEM at largest block size (float)
      'err_naive'        : SEM at block size 1 (float)
      'plateau_reached'  : bool
      'block_sizes'      : list of block sizes used
      'block_sems'       : list of block-SEM values
      'n_blocks'         : list of block counts per level
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 * min_blocks:
        sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return {
            "err": sem,
            "err_lastblock": sem,
            "err_naive": sem,
            "plateau_reached": False,
            "block_sizes": [1],
            "block_sems": [sem],
            "n_blocks": [int(n)],
        }

    sizes, sems, nblocks = [], [], []
    cur = x.copy()
    bsize = 1
    while len(cur) >= min_blocks:
        sem = float(cur.std(ddof=1) / np.sqrt(len(cur)))
        sizes.append(bsize); sems.append(sem); nblocks.append(int(len(cur)))
        # pairwise block average
        m = (len(cur) // 2) * 2
        cur = 0.5 * (cur[:m:2] + cur[1:m:2])
        bsize *= 2

    err_naive = sems[0]
    err_lastblock = sems[-1]

    # Plateau detection: last `plateau_window` points agree within plateau_tol.
    plateau = False
    if len(sems) >= plateau_window:
        tail = np.asarray(sems[-plateau_window:])
        if tail.min() > 0 and (tail.max() - tail.min()) / tail.min() <= plateau_tol:
            plateau = True
            err_report = float(tail.mean())
        else:
            err_report = float(max(sems))
    else:
        err_report = float(max(sems))

    # The reported error is at least the naive SEM.
    err_report = max(err_report, err_naive)

    return {
        "err": err_report,
        "err_lastblock": err_lastblock,
        "err_naive": err_naive,
        "plateau_reached": plateau,
        "block_sizes": sizes,
        "block_sems": sems,
        "n_blocks": nblocks,
    }


# ── High-level interface ────────────────────────────────────────────────

def compute_pressure_from_run(run_dir, max_configs=50, skip_configs=5,
                              year=1979, progress=True, wl_glob=None):
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
    progress : bool
        Print a progress bar to stderr.
    wl_glob : str or None
        Glob pattern for worldline files. Relative paths resolved against
        run_dir. Default: OUTPUT/ce-wl-*.dat.

    Returns
    -------
    dict with keys:
        P_mean_bar, P_err_bar : mean pressure and standard error (bar)
        P_tail_bar : tail correction (bar)
        P_total_bar : P_mean + P_tail (bar)
        n_configs : number of configs processed
        components : dict with P_ideal, P_spring, P_virial means (bar)
    """
    import sys

    params_file = os.path.join(run_dir, "params.json")
    with open(params_file) as pf:
        params = json.load(pf)

    N = params["N"]
    tau = params["dt"]
    L = params.get("L")
    Lx = params.get("Lx", L)
    Ly = params.get("Ly", L)
    Lz = params.get("Lz", L)
    side = np.array([Lx, Ly, Lz])
    rho = params.get("rho", params.get("n", N / (Lx * Ly * Lz)))
    rc = params["potential_cutoff"]
    lam = HBAR2_OVER_2M_HE4

    dvdr_fn = lambda r: _dvdr_numpy(r, year=year)

    if wl_glob is None:
        pattern = os.path.join(run_dir, "OUTPUT", "ce-wl-*.dat")
    elif not os.path.isabs(wl_glob):
        pattern = os.path.join(run_dir, wl_glob)
    else:
        pattern = wl_glob
    wl_files = sorted(globmod.glob(pattern))
    if not wl_files:
        raise FileNotFoundError(f"No worldline files matching {pattern}")

    # M: from params, or auto-detect from the first config in the worldline file
    M = params.get("M")
    if M is None:
        with open(wl_files[0]) as _f:
            _slices = set()
            _in = False
            for _line in _f:
                if '# START_CONFIG' in _line:
                    _slices = set()
                    _in = True
                elif '# END_CONFIG' in _line:
                    if _in and _slices:
                        M = len(_slices)
                        break
                    _in = False
                elif _in and not _line.startswith('#'):
                    parts = _line.split()
                    if len(parts) >= 10:
                        _slices.add(int(parts[0]))
        if M is None:
            raise ValueError("Cannot determine M from params.json or worldline data")
        if progress:
            print(f"  Auto-detected M={M} from worldline data", file=sys.stderr)

    P_totals = []
    P_components = []

    run_label = os.path.basename(run_dir)

    for i, (pos, ns, np_arr) in enumerate(parse_wl_configs(
            wl_files[0], N, M,
            max_configs=max_configs, skip_configs=skip_configs)):
        P_id, P_sp, P_vir = compute_pressure_config(
            pos, ns, np_arr, N, M, side, tau, lam, rc, dvdr_fn)
        P_totals.append(P_id + P_sp + P_vir)
        P_components.append((P_id, P_sp, P_vir))
        if progress:
            done = i + 1
            bar_len = 30
            filled = int(bar_len * done / max_configs)
            bar = '#' * filled + '-' * (bar_len - filled)
            p_running = np.mean(P_totals) * K_A3_TO_BAR
            sys.stderr.write(
                f"\r    P [{bar}] {done}/{max_configs} "
                f"({run_label})  P={p_running:.1f} bar")
            sys.stderr.flush()

    if progress and P_totals:
        sys.stderr.write('\n')
        sys.stderr.flush()

    if not P_totals:
        raise ValueError(f"No diagonal configs found in {run_dir}")

    P_totals = np.array(P_totals)
    comps = np.array(P_components)
    n_cfg = len(P_totals)

    P_tail = _tail_pressure(rho, rc, year=year)

    # Per-config arrays in bar
    P_total_per_config = P_totals * K_A3_TO_BAR
    P_ideal_per_config = comps[:, 0] * K_A3_TO_BAR
    P_spring_per_config = comps[:, 1] * K_A3_TO_BAR
    P_virial_per_config = comps[:, 2] * K_A3_TO_BAR

    # Error analysis: Flyvbjerg-Petersen blocking is the reported error
    # (P_err_bar). We also keep the naive SEM and the ACF-based tau_int for
    # diagnostic comparison. If the blocking curve has not plateaued, the
    # reported error is the maximum observed block SEM (a lower bound on
    # the true SEM), and `P_err_plateau_reached` is False — callers should
    # treat the number as "error floor", not as a converged estimate.
    P_bar_arr = P_totals * K_A3_TO_BAR
    naive_err = float(P_bar_arr.std(ddof=1) / np.sqrt(n_cfg))
    tau_int, corr_err = _correlated_error(P_bar_arr)
    blk = _blocking_error(P_bar_arr)

    return {
        'P_mean_bar': float(P_totals.mean() * K_A3_TO_BAR),
        'P_err_bar': blk["err"],
        'P_err_naive_bar': naive_err,
        'P_err_acf_bar': corr_err,
        'P_err_lastblock_bar': blk["err_lastblock"],
        'P_err_plateau_reached': bool(blk["plateau_reached"]),
        'P_block_sizes': blk["block_sizes"],
        'P_block_sems_bar': blk["block_sems"],
        'P_tau_int': tau_int,
        'P_tail_bar': float(P_tail * K_A3_TO_BAR),
        'P_total_bar': float((P_totals.mean() + P_tail) * K_A3_TO_BAR),
        'n_configs': n_cfg,
        'components': {
            'P_ideal_bar': float(comps[:, 0].mean() * K_A3_TO_BAR),
            'P_spring_bar': float(comps[:, 1].mean() * K_A3_TO_BAR),
            'P_virial_bar': float(comps[:, 2].mean() * K_A3_TO_BAR),
        },
        'per_config': {
            'P_total_bar': P_total_per_config,
            'P_ideal_bar': P_ideal_per_config,
            'P_spring_bar': P_spring_per_config,
            'P_virial_bar': P_virial_per_config,
        },
    }
