"""
jax_landscape: A JAX-MD-based package for molecular dynamics simulations.
"""

__version__ = "0.1.0"

from . import potentials  # noqa: F401  — always available

try:
    from .energy_fun import (
        aziz_1995,
        build_energy_fn_aziz_1995_no_neighborlist,
        build_energy_fn_aziz_1995_neighborlist
    )
except ImportError:
    # jax-md not installed — potentials subpackage still works
    pass

__all__ = [
    "potentials",
    "aziz_1995",
    "build_energy_fn_aziz_1995_no_neighborlist",
    "build_energy_fn_aziz_1995_neighborlist"
]
