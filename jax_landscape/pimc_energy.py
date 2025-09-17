"""PIMC total potential energy U_RP (i.e., path probability weight over beta) for a single (closed) worldline configuration. The energy function here follows Eq.2.29 of Ceperley 1995 (Rev. Mod. Phys. 67, 279). The worldline configuration must be the Path object, compatible with the PIMC code of Adrian, which allows particle permutation at each time slice. See more details in notes [TODO: xxx].  

The returned function is JAX-jittable because internally only JAX arrays are used.
"""
from __future__ import annotations
from typing import Callable, Dict
import jax
import jax.numpy as jnp

try:  # type hint import (optional)
    from .io.pimc import Path  # noqa: F401
except Exception:  # pragma: no cover
    Path = object  # fallback for type checkers

Array = jnp.ndarray


def _validate_closed_worldline(next_indices: Array, M: int, N: int) -> None:
    """Validate that all beads form closed cycles with length multiple of M.
    TODO: check that it is correct and not redundant with other files.
    Args:
        next_indices: int array (M,N,2) giving (m', j) for each forward link.
        M, N: dimensions.
    Raises:
        ValueError: if any cycle is open or has length not divisible by M.
    """
    # We do this in Python (once per configuration) because it involves graph traversal.
    visited = [[False]*N for _ in range(M)]
    for m in range(M):
        for i in range(N):
            if visited[m][i]:
                continue
            # Traverse cycle starting at (m,i)
            length = 0
            cur_m, cur_i = m, i
            while not visited[cur_m][cur_i]:
                visited[cur_m][cur_i] = True
                nxt = next_indices[cur_m, cur_i]
                cur_m, cur_i = int(nxt[0]), int(nxt[1])
                length += 1
                if length > M * N:  # safeguard
                    raise ValueError("Detected non-terminating cycle (possible open chain).")
            # Completed a cycle; length must be a multiple of M
            if length % M != 0:
                raise ValueError(
                    f"Invalid worldline cycle length {length}; not a multiple of M={M}. Worldline must be closed."
                )


def _total_spring_term(bead_coords: Array, next_indices: Array, beta: float, hbar: float, mass: float, displacement_fn: Callable[[Array, Array], Array]) -> Array:
    """Compute total spring (polymer) energy:
         E_spring = 1/2 * (m nb / (beta hbar)^2) * Σ_{links} |Δr_link|^2
        where the sum is over all links in the worldlines, regardless of particle labels or cycles. See my notes [TODO: xxx].
    Args:
        bead_coords: (M,N,3)
        next_indices: (M,N,2)
        beta: inverse temperature β
        hbar: reduced Planck constant ℏ
        mass: particle mass m
        displacement_fn: periodic displacement function mapping (r1,r2)->Δr
    Returns:
        Scalar spring energy.
    """
    M, N, _ = bead_coords.shape
    next_m = next_indices[..., 0]
    next_i = next_indices[..., 1]
    forward_coords = bead_coords[next_m, next_i]
    r1 = bead_coords.reshape(-1, 3)
    r2 = forward_coords.reshape(-1, 3)
    # Use vmap over individual bead pairs because space.displacement expects vector inputs.
    dr = jax.vmap(displacement_fn)(r1, r2)
    dr2 = jnp.sum(dr * dr, axis=-1)
    sum_dr2 = jnp.sum(dr2)
    return 0.5 * mass * M / ((beta * hbar)**2) * sum_dr2


def _total_potential_term(bead_coords: Array, potential_energy_fn: Callable[[Array], Array]) -> Array:
    """Average potential energy over time slices. 
        V = (1 / M) * Σ_{m=0}^{M-1} V_classical(R[m])

    Args:
        bead_coords: (M,N,3)
        potential_energy_fn: maps (N,3) -> scalar. This is the potential energy function for classical system.
    Returns:
        Scalar average potential energy V.
    """
    def per_slice(R_slice):
        return potential_energy_fn(R_slice)
    V_slices = jax.vmap(per_slice)(bead_coords)  # (M,)
    return jnp.mean(V_slices)


def build_pimc_energy_fn(displacement_fn: Callable[[Array, Array], Array], potential_energy_fn: Callable[[Array], Array]):
    """Factory producing a energy function U_RP for a single PIMC configuration. This is the weight/beta of the path integral.

    Parameters:
        displacement_fn: JAX-MD displacement function (periodic or free space)
        potential_energy_fn: Single-slice potential energy function mapping (N,3)->scalar. This is the energy function for a classical configuration.

    Returns:
        Function `pimc_energy_fn(path, beta, hbar, mass = 1.0)` -> dict(E=..., K=..., V=...)
    """

    def pimc_energy_fn(path, beta: float, hbar: float, mass: float = 1.0) -> Dict[str, Array]:
        # Extract arrays (convert to jnp)
        bead_coords = jnp.asarray(path.beadCoord)
        next_indices = jnp.asarray(path.next)
        M, N, _ = bead_coords.shape

        # TODO: move to when Path initialized not here. 
        # Validate closed worldline (Python side; raises early if invalid)
        # if not getattr(path, 'is_closed_worldline', False):
        #     raise ValueError("Worldline must be closed (path.is_closed_worldline False).")
        # _validate_closed_worldline(path.next, M, N)

        Esp = _total_spring_term(bead_coords, next_indices, beta, hbar, mass, displacement_fn)
        Eint = _total_potential_term(bead_coords, potential_energy_fn)
        Urp = Esp + Eint
        return {'Urp': Urp, 'E_sp': Esp, 'E_int': Eint} # RP potential energy, spring energy, interaction potential energy.   'E_qm': 0.0, 

    return pimc_energy_fn

__all__ = [
    'build_pimc_energy_fn'
]
