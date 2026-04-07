#!/usr/bin/env python
"""
Generate C++-equivalent Aziz reference values using the exact formulas
from programs/pimc/src/potential.cpp and programs/pimc/include/potential.h.

This uses plain Python math (no JAX) to produce an independent reference
for testing the JAX implementation.
"""

import json
import math

AZIZ_PARAMS = {
    1979: {
        'A': 0.5448504e6, 'alpha': 13.353384, 'beta': 0.0,
        'C6': 1.3732412, 'C8': 0.4253785, 'C10': 0.1781,
        'D': 1.241314, 'epsilon': 10.8, 'rm': 2.9673,
    },
    1987: {
        'A': 1.8443101e5, 'alpha': 10.43329537, 'beta': -2.27965105,
        'C6': 1.36745214, 'C8': 0.42123807, 'C10': 0.17473318,
        'D': 1.4826, 'epsilon': 10.948, 'rm': 2.9673,
    },
    1995: {
        'A': 1.86924404e5, 'alpha': 10.5717543, 'beta': -2.07758779,
        'C6': 1.35186623, 'C8': 0.41495143, 'C10': 0.17151143,
        'D': 1.438, 'epsilon': 10.956, 'rm': 2.9683,
    },
}

R_GRID = [1.5, 2.0, 2.5, 2.9673, 3.0, 3.5, 4.0, 5.0, 7.0, 9.0]


def F(x, D):
    """Damping function — C++ potential.h:709-711"""
    if x < D:
        return math.exp(-(D / x - 1.0)**2)
    return 1.0


def dF(x, D):
    """Derivative of damping — C++ potential.h:714-717"""
    if x < D:
        ix = 1.0 / x
        return 2.0 * D * ix * ix * (D * ix - 1.0) * math.exp(-(D * ix - 1.0)**2)
    return 0.0


def valueV(r, p):
    """C++ AzizPotential::valueV — potential.cpp:1660-1679"""
    x = r / p['rm']
    Urep = p['A'] * math.exp(-p['alpha'] * x + p['beta'] * x * x)

    if x < 0.01:
        return p['epsilon'] * Urep

    ix2 = 1.0 / (x * x)
    ix6 = ix2 * ix2 * ix2
    ix8 = ix6 * ix2
    ix10 = ix8 * ix2
    Uatt = -(p['C6'] * ix6 + p['C8'] * ix8 + p['C10'] * ix10) * F(x, p['D'])
    return p['epsilon'] * (Urep + Uatt)


def valuedVdr(r, p):
    """C++ AzizPotential::valuedVdr — potential.cpp:1686-1712"""
    x = r / p['rm']
    T1 = p['A'] * (-p['alpha'] + 2.0 * p['beta'] * x) * \
         math.exp(-p['alpha'] * x + p['beta'] * x * x)

    if x < 0.01:
        return (p['epsilon'] / p['rm']) * T1

    ix = 1.0 / x
    ix2 = ix * ix
    ix6 = ix2 * ix2 * ix2
    ix7 = ix6 * ix
    ix8 = ix6 * ix2
    ix9 = ix8 * ix
    ix10 = ix8 * ix2
    ix11 = ix10 * ix
    T2 = (6.0 * p['C6'] * ix7 + 8.0 * p['C8'] * ix9 + 10.0 * p['C10'] * ix11) * F(x, p['D'])
    T3 = -(p['C6'] * ix6 + p['C8'] * ix8 + p['C10'] * ix10) * dF(x, p['D'])
    return (p['epsilon'] / p['rm']) * (T1 + T2 + T3)


def tailV(rc, p):
    """C++ tail correction — potential.cpp:1640-1645"""
    rmorc = p['rm'] / rc
    t2 = p['C6'] * rmorc**3 / 3.0
    t3 = p['C8'] * rmorc**5 / 5.0
    t4 = p['C10'] * rmorc**7 / 7.0
    return 2.0 * math.pi * p['epsilon'] * (-p['rm']**3 * (t2 + t3 + t4))


def main():
    reference = {}
    for year in [1979, 1987, 1995]:
        p = AZIZ_PARAMS[year]
        year_data = {
            'params': p,
            'r_grid': R_GRID,
            'V': [valueV(r, p) for r in R_GRID],
            'dVdr': [valuedVdr(r, p) for r in R_GRID],
            'tailV_rc_9': tailV(9.0, p),
        }
        reference[str(year)] = year_data

    with open('aziz_cpp_reference.json', 'w') as f:
        json.dump(reference, f, indent=2)
    print("Written aziz_cpp_reference.json")


if __name__ == '__main__':
    main()
