# JAX-Landscape

Energy landscape analysis toolkit for molecular systems using JAX.

## Overview

JAX-Landscape provides tools for analyzing potential energy landscapes of quantum and classical molecular systems. The toolkit supports Path Integral Monte Carlo (PIMC) configurations and classical molecular structures, with efficient energy calculations, local minimization, and Hessian analysis, and more...

Built on JAX for automatic differentiation and performance optimization.

## Features

- **Unified interface** for classical and PIMC configurations
- **Force fields** includes Aziz 1995 potential for Helium interactions or any user defined forms
- **PIMC plugin** for path integral simulations
- **Local minimization** with saddle point detection and escape
- **Hessian eigenanalysis** for characterizing stationary points

## Installation

```bash
git clone https://github.com/yourusername/jax-landscape.git
cd jax-landscape

python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -e ".[dev]"  # For development and testing
```

## Quick Start

### Find local minima

```python
import jax
import jax.numpy as jnp
from jax_md import space
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_no_neighborlist
from jax_landscape.local_minima import find_local_minimum

jax.config.update("jax_enable_x64", True)

# Setup
box_size = jnp.array([20.0, 20.0, 20.0])  # Angstrom
displacement_fn, _ = space.periodic(box_size)
energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)

# Initial configuration
xyz_initial = jnp.array([[0.0, 0.0, 0.0],
                         [3.0, 0.0, 0.0]])

# Minimize
result = find_local_minimum(energy_fn, xyz_initial)
print(f"Energy: {result['energy_initial']:.6f} → {result['energy_final']:.6f}")
```


### Compute Hessian spectrum

```python
from jax_landscape.hessian_eigenvals import compute_hessian_eigenvalues

result = compute_hessian_eigenvalues(energy_fn, xyz_minimum)
print(f"Lowest eigenvalues: {result['eigenvalues'][:6]}")
```

## Core Modules

### `energy_fun.py`
Compute total energy of the system.
- Force fields include: Aziz 1995 potential
- Easy to define new force fields.

### `pimc_energy.py`
PIMC total energy (U_RP) following Ceperley 1995.

- Supports closed worldline configurations with boson permutation at each time slice.

### `local_minima.py`
Gradient-based local minimization using scipy with JAX gradients.

- `find_local_minimum(energy_fn, xyz_initial, ...)` - Main minimization routine
- Saddle point detection via Hessian eigenanalysis
- Automatic saddle escape along negative curvature directions
- Optional trajectory saving in PIMC worldline format

### `hessian_eigenvals.py`
Hessian computation via JAX autodiff.

- `compute_hessian_eigenvalues(energy_fn, xyz, ...)` - Full Hessian eigenanalysis
- Returns eigenvalues, eigenvectors, and optionally the full Hessian matrix

### `io/pimc.py`
PIMC worldline file I/O.

- `load_pimc_worldline_file(path, Lx, Ly, Lz)` - Load configurations
- `Path` class - Worldline representation with connectivity

## Testing

The test suite validates numerical accuracy, mathematical properties, and edge cases.

```bash
pytest                              # Run all tests
pytest tests/test_energy_fun.py -v  # Specific module
pytest --cov=jax_landscape          # With coverage
```

Here is the test coverage overview by Module:

### `energy_fun.py` - Energy Calculations
**Covered:**
- Classical systems: N=6, N=500
- Both neighbor list and no-neighbor-list implementations
- Energy and gradient accuracy (< 1e-8 relative error)
- Periodic boundary conditions

**NOT Covered:**
- Cutoff radius edge cases
- Non-cubic boxes
- Neighbor list updates during dynamics

---

### `pimc_energy.py` - PIMC Energy Functions
**Covered:**
- Classical limit: N=3, M=1 (spring energy = 0)
- PIMC system with zero interaction: N=2, M=3, 1-cycle and 2-cycle configurations
- PIMC system with Aziz potential: N=64, T=1.55K, n=0.0218 Å⁻³


**Methods**
- Compare with calculations from independent code
- [TODO] Compare with analytic results

**NOT Covered:**
- Different temperatures, masses, hbar values
- PIMC energy gradients

---

### `local_minima.py` - Local Minimization
**Covered:**
- Classical: N=6 Aziz system
- [TODO] Classical: N=512 Aziz system
- PIMC: N=2, M=3, one cycle
- PIMC: N=2, M=3, M=100, two-cycle
- PIMC: N=64. See `example/pimc_minimize`
  - normal liquid  
  - superfluid

- Optimizer: trust-ncg with escape_saddles=True

**Methods**
- To ensure it is a local minima, we use 4-property validation: zero gradient, energy decrease, local stability, Hessian eigenvalues
- To ensure the minimizaiton remains in the same basin, we monitor the minimization trajectory.

**NOT Covered:**
- Other optimization methods: L-BFGS-B, CG, Newton-CG

---

### `hessian_eigenvals.py` - Hessian Analysis
**Covered:**
- Classical: N=6 Aziz system
- PIMC: N=2, M=3 free particle
- Analytical validation: 2-particle harmonic oscillator

**Methods**
- Numerical consistency: autodiff vs finite differences
- Eigenvalue/eigenvector reconstruction (H·v = λ·v)

**NOT Covered:**
- [TODO] Large systems: N=500 (requires large memory)
- PIMC with interactions (Aziz potential Hessian)
- Ill-conditioned Hessians
- Eigenvector orthogonality verification

---

### `io/pimc.py` - PIMC I/O
**Covered:**
- File loading: N=2, M=3
- Path object construction (beadCoord, next, prev arrays)
- Cycle detection and size distribution
- Connectivity validation

**NOT Covered:**
- File writing (`save_pimc_worldline_file`)
- Multiple configurations in one file
- Large systems: N=64, M>3
- Invalid/malformed files
- Round-trip test: save → load → compare


## References

- Aziz, R. A. "A new determination of the ground state interatomic potential for He2." *Mol. Phys.* **61**, 1487 (1987).
- Ceperley, D. M. "Path integrals in the theory of condensed helium." *Rev. Mod. Phys.* **67**, 279 (1995).

## License

[To be added]
