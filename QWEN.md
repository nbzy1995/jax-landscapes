# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

General-purpose JAX energy-landscape analysis library. Local minimization,
saddle-point detection and escape, and Hessian analysis accept user-defined
differentiable energy functions. Path-integral energy construction and worldline
I/O extend these tools to molecular systems. Helium with the Aziz 1995 potential
is a worked example and validation case, not the scope of the library.

## Environment Setup

Python virtual environment is located at `.venv/` in the project root.

Use `uv sync --frozen --extra dev --extra demo` for the checked environment.
Add `--extra notebooks` for the existing interactive notebooks.
Dependencies and package discovery are configured in `pyproject.toml`; `uv.lock`
records the resolved versions.

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
- `jax_landscape/io/pimc.py`: Loader/writer for PIMC worldline files (`ce-wl-*.dat`)
- `example/helium_exchange.py`: reproducible two-particle exchange demonstration

### Examples

- `example/pimc_visualization/`: Interactive 3D visualization of PIMC worldlines
  - `vis-worldlines.ipynb`: Plotly-based visualization notebook
  - Works with both PIMC output and minimization trajectories
  - Requires: plotly, ipywidgets

#### External Dependencies

- JAX-MD: pinned fork commit in `pyproject.toml`; it supplies the OpenMM-style C2 cutoff.
- JAX/JAXLIB: 0.4.38. The older fork does not import with JAX 0.9.0.1.
- See `docs/validation.md` for numerical evidence and the unresolved PIMC fixture.

## Demonstration

Run `uv run --frozen python example/helium_exchange.py` from the repository root.
The package has no `jax_landscape.main` module or installed CLI. Use the Python
API and the example script documented in `README.md`.

## Unit System

The Aziz 1995 potential uses:
- Distance: Ångström
- Energy: kB·K (Boltzmann constant × Kelvin)

Test data uses nm and kJ/mol, with conversions:
- `nm_to_Angstrom = 10.0`
- `KJmol_to_KBK = 120.2722922542`

## JAX Configuration

The codebase requires float64 precision. The example and test files set:

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

`find_local_minimum()` provides a unified interface for minimizing any energy function:
- Takes pre-built `energy_fn(xyz)` directly (not factory)
- **WARNING: Do NOT use energy functions with neighbor lists during minimization**
  - Particles can move outside the neighbor list cell during optimization
  - This causes incorrect energy/gradient calculations
  - Always use `build_energy_fn_aziz_1995_no_neighborlist()` for minimization
- Supports classical and PIMC energy functions via same interface
- Returns dict with `xyz_final`, `energy_final`, `energy_initial`, convergence info
- Can log optimization progress to file

**Usage**:
```python
# Classical minimization (always use no_neighborlist version)
displacement_fn, _ = space.periodic(box_size)
energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
result = find_local_minimum(energy_fn, xyz_initial)

# PIMC with trajectory saving
from jax_landscape.pimc_energy import build_pimc_energy_fn_xyz
pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
    pimc_energy_fn, path_obj, beta, hbar, mass
)
result = find_local_minimum(
    minimization_energy_fn,
    path_obj.beadCoord,
    trajectory_file='trajectory.dat',
    trajectory_path_template=path_template,
    save_trajectory_every=10
)
```

**PIMC Minimization**: Use `build_pimc_energy_fn_xyz()` to create a wrapper that:
- Maintains fixed worldline connectivity (next/prev arrays)
- Only optimizes bead coordinates
- Returns (energy_fn, path_template) tuple
- path_template enables trajectory saving in PIMC worldline format

**Trajectory Saving**: For PIMC minimization, snapshots can be saved in worldline format:
- Output file is identical to PIMC worldline files (`ce-wl-*.dat`)
- Each configuration corresponds to a minimization iteration
- Can be visualized using existing PIMC visualization tools
- Connectivity is preserved across all snapshots

## Code Style

- Line length: 88 characters (Black)
- Python: 3.11+
- Type hints encouraged but not enforced
- Import sorting: isort with Black profile