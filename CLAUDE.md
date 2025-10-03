# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based energy landscape analysis toolkit for quantum systems, with a focus on Path Integral Monte Carlo (PIMC) simulations and classical energy calculations. The codebase uses JAX-MD for molecular dynamics primitives and implements the Aziz 1995 potential for Helium interactions.

## Environment Setup

Python virtual environment is located at `.venv/` in the project root.

Activate: `source .venv/bin/activate`

Install package in editable mode: `pip install -e .`

Install dev dependencies: `pip install -e ".[dev]"`

## Testing

Run all tests: `pytest`

Run specific test file: `pytest tests/test_energy_fun.py`

Run single test: `pytest tests/test_energy_fun.py::test_energy_grad -v`

Run tests in parallel: `pytest -n auto`

## Project Structure

### Core Modules

- `jax_landscape/energy_fun.py`: Aziz 1995 potential implementation with/without neighbor lists
- `jax_landscape/pimc_energy.py`: PIMC total energy (U_RP) for closed worldline configurations
- `jax_landscape/local_minima.py`: Local minimization using scipy.optimize with JAX gradients
- `jax_landscape/io/pimc.py`: Loader for PIMC worldline output files (`ce-wl-*.dat`)
- `jax_landscape/main.py`: CLI entry point supporting classical and PIMC input formats

### External Dependencies

- `external/jax-md/`: Modified JAX-MD library included in the repository

## Command Line Interface

The main entry point is `jax_landscape` (installed via `setup.py`):

```bash
# Classical energy calculation
jax_landscape --input_file <path.json> --input_format classical

# PIMC configuration loading
jax_landscape --input_file <ce-wl-*.dat> --input_format pimc --config_index 0

# With minimization
jax_landscape --input_file <path> --run_minimize True
```

## Unit System

The Aziz 1995 potential uses:
- Distance: Ångström
- Energy: kB·K (Boltzmann constant × Kelvin)

Test data uses nm and kJ/mol, with conversions:
- `nm_to_Angstrom = 10.0`
- `KJmol_to_KBK = 120.2722922542`

## JAX Configuration

The codebase requires float64 precision. Both `main.py` and test files set:

```python
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")
```

## Architecture Notes

### Energy Function Factories

Energy functions are created via factory functions that return JAX-jittable energy functions:

1. `build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)` - Direct pair summation
2. `build_energy_fn_aziz_1995_neighborlist(displacement_fn, box_size)` - Returns `(neighbor_fn, energy_fn)` tuple
3. `build_pimc_energy_fn(displacement_fn, potential_energy_fn)` - PIMC ring-polymer energy

### PIMC Worldline Structure

The `Path` class represents a single PIMC configuration with:
- `beadCoord`: shape `(M, N, 3)` where M=time slices, N=particles
- `next`, `prev`: connectivity arrays shape `(M, N, 2)` storing `(time_slice_idx, particle_idx)`
- Support for closed worldlines with cycle detection
- Compatibility with boson permutation at each time slice

### Local Minimization

`find_local_minimum()` uses scipy.optimize.minimize with JAX-computed gradients. The function:
- Automatically handles neighbor list reallocation during optimization
- Supports different energy function factories via factory pattern
- Returns dict with `xyz_final`, `energy_final`, `energy_initial`, convergence info
- Can log optimization progress to file

## Code Style

- Line length: 88 characters (Black)
- Python: 3.11+
- Type hints encouraged but not enforced
- Import sorting: isort with Black profile