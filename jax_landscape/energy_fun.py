"""
Energy functions for Helium interactions using JAX-MD.

The bare pair potential lives in ``jax_landscape.potentials.aziz``.
This module applies switching/cutoff via JAX-MD and builds energy
functions suitable for minimization.
"""

import numpy as np
import jax.numpy as jnp
from jax_md import energy, space, smap, partition

from .potentials.aziz import V as _aziz_V, AZIZ_PARAMS


def _validate_cutoff_against_box(box_size, r_cutoff, context, enforce=True):
    """
    Ensure the cutoff radius is compatible with the minimum-image convention: r_cutoff <= 0.5 * min(box_size).
    """
    if not enforce or box_size is None:
        return

    box = np.asarray(box_size, dtype=float)
    if box.ndim == 0:
        box = np.repeat(box, 3)

    min_box = float(np.min(box))

    max_allowed = 0.5 * min_box
    if r_cutoff > max_allowed + 1e-9:
        raise ValueError(
            f"{context}: cutoff {r_cutoff:.4f} Å exceeds half of the "
            f"smallest box length ({max_allowed:.4f} Å). "
            "Increase the simulation cell or reduce the cutoff."
        )


# ── Pair potential wrappers (JAX-MD compatible) ──────────────────────────

def aziz_1995(r, **kwargs):
    """Aziz 1995 pair potential (backward-compatible wrapper)."""
    return _aziz_V(r, year=1995)


def _build_aziz_pair_fn(year=1979):
    """Return a pair function V(r) for the given Aziz year."""
    def pair_fn(r, **kwargs):
        return _aziz_V(r, year=year)
    return pair_fn


# ── Generic factory ──────────────────────────────────────────────────────

def build_energy_fn_aziz(
    displacement_or_metric,
    year=1979,
    r_cutoff=13.6,
    r_sw=None,
    box_size=None,
    enforce_cutoff=True,
    use_neighborlist=False,
    dr_threshold=0.5,
    nl_format=partition.OrderedSparse,
    **kwargs):
    """
    Build a JAX-MD energy function using the Aziz potential.

    Parameters
    ----------
    displacement_or_metric : callable
        JAX-MD displacement function.
    year : int
        Aziz parameterization year (1979, 1987, 1995).
    r_cutoff : float
        Pair cutoff radius in Angstrom.
    r_sw : float or None
        Switching distance. If None, defaults to 0.9 * r_cutoff.
        Set r_sw = r_cutoff to disable switching (validation mode).
    box_size : array-like or None
        Box dimensions for cutoff validation.
    enforce_cutoff : bool
        If True, validate r_cutoff <= L/2.
    use_neighborlist : bool
        If True, return (neighbor_fn, energy_fn) tuple.
    dr_threshold : float
        Neighbor list buffer (only used if use_neighborlist=True).
    nl_format : partition format
        Neighbor list format (only used if use_neighborlist=True).

    Returns
    -------
    energy_fn or (neighbor_fn, energy_fn)
    """
    if r_sw is None:
        r_sw = 0.9 * r_cutoff

    _validate_cutoff_against_box(
        box_size, r_cutoff,
        f"build_energy_fn_aziz(year={year})",
        enforce=enforce_cutoff
    )

    pair_fn = _build_aziz_pair_fn(year)
    r_cutoff = jnp.array(r_cutoff, jnp.float64)
    r_sw = jnp.array(r_sw, jnp.float64)

    cutoff_pair = energy.multiplicative_isotropic_cutoff(pair_fn, r_sw, r_cutoff)
    metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

    if not use_neighborlist:
        return smap.pair(cutoff_pair, metric)

    dr_threshold = jnp.array(dr_threshold, jnp.float64)
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric, box_size, r_cutoff,
        dr_threshold=dr_threshold, format=nl_format)
    energy_fn = smap.pair_neighbor_list(cutoff_pair, metric)
    return neighbor_fn, energy_fn


# ── Backward-compatible wrappers ─────────────────────────────────────────

def build_energy_fn_aziz_1995_no_neighborlist(
    displacement_or_metric,
    r_cutoff=13.6,
    r_sw=13.6*0.9,
    box_size=None,
    enforce_cutoff=True,
    **kwargs):
    """Backward-compatible wrapper: Aziz 1995 without neighbor list."""
    return build_energy_fn_aziz(
        displacement_or_metric, year=1995,
        r_cutoff=r_cutoff, r_sw=r_sw,
        box_size=box_size, enforce_cutoff=enforce_cutoff,
        use_neighborlist=False)


def build_energy_fn_aziz_1995_neighborlist(
    displacement_or_metric,
    box_size,
    r_cutoff=13.6,
    r_sw=13.6*0.9,
    dr_threshold=0.5,
    format=partition.OrderedSparse,
    enforce_cutoff=True,
    **kwargs):
    """Backward-compatible wrapper: Aziz 1995 with neighbor list."""
    return build_energy_fn_aziz(
        displacement_or_metric, year=1995,
        r_cutoff=r_cutoff, r_sw=r_sw,
        box_size=box_size, enforce_cutoff=enforce_cutoff,
        use_neighborlist=True,
        dr_threshold=dr_threshold, nl_format=format)
