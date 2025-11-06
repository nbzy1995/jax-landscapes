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

### Test Coverage

See [TEST_COVERAGE.md](TEST_COVERAGE.md) for detailed coverage matrix.

## Units

- **Distance**: Ångström (Å)
- **Energy**: kB·K (Boltzmann constant × Kelvin)
- **Mass**: Atomic mass units (amu)

## References

- Aziz, R. A. "A new determination of the ground state interatomic potential for He2." *Mol. Phys.* **61**, 1487 (1987).
- Ceperley, D. M. "Path integrals in the theory of condensed helium." *Rev. Mod. Phys.* **67**, 279 (1995).

## License

[To be added]
