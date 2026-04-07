"""
Pair potentials for helium simulations.

Submodules:
    aziz — Aziz He-He pair potential (1979, 1987, 1995 parameterizations)
"""

from .aziz import V, V_vec, dVdr, dVdr_vec, get_params, tail_V, tail_pressure
