# Validation and reproducibility

## Two-particle exchange example

`example/helium_exchange.py` uses the public library APIs to construct and minimize
one synthetic closed exchange cycle. It does not run a Monte Carlo sampler.

| Parameter | Value |
|---|---|
| Particles / imaginary-time slices | 2 / 16 |
| Boundary conditions | Free space |
| Temperature | 1 K |
| Coordinates / energy | Å / kB·K |
| Mass unit | Helium-4 mass; dimensionless mass = 1 |
| Reduced ℏ | 3.4812756477 |
| Initial geometry | Perturbed elliptical ring, seed 7 |
| Pair potential | Aziz HFD-B3-FCI1; switching onset 12.24 Å, cutoff 13.6 Å |
| Optimizer | `trust-ncg`; gradient tolerance 10⁻⁶; maximum 500 iterations |
| Curvature check | Full Hessian; negative-eigenvalue threshold −10⁻⁶ |

The fixed connectivity joins the two particles into one exchange cycle. Only
coordinates vary during optimization. The ring-polymer objective is the sum of
the harmonic link term and the interaction energy averaged over slices.

The checked run used JAX 0.4.38, float64 and the CPU backend:

- Energy: 0.2665875748 → −9.1753807113 kB·K.
- 21 iterations; no saddle-escape step required.
- Final gradient norm: 3.3246804 × 10⁻⁷.
- Smallest Hessian eigenvalue: −6.82158 × 10⁻⁹; none below −10⁻⁶.

Near-zero eigenvalues are expected from rigid-motion symmetries. These are
numerical stationarity and curvature checks at stated tolerances, not a proof of
uniqueness or global optimality. The example raises an error if its convergence,
gradient or curvature checks fail. Animation frames are recorded optimizer states,
with no interpolation of coordinates.

[Machine-readable result](media/result.json). Reproduce with:

```sh
uv run --frozen python example/helium_exchange.py
```

## Saved OpenMM comparison

The [preserved output](openmm-comparison.txt) is an excerpt of the existing
`example/pimc_He_minimize/compare_openmm/run.log`, starting at the comparison stage.
The comparison code is [compare_results.py](../example/pimc_He_minimize/compare_openmm/compare_results.py).
It reports a 64-particle, 100-slice calculation at 2.5 K, in a 14.32103 Å box.

The comparator converts the OpenMM summed energy to the JAX slice-averaged
convention and reports:

- Initial total, spring and interaction energies: pass at 0.1% relative tolerance.
- Final total and interaction energies: pass at the same tolerance.
- Final spring energy: both values round to 0.00. The comparator reports zero
  relative error whenever the JAX magnitude is at or below 10⁻¹⁰, so this entry does
  not establish agreement at the stated relative tolerance.
- Final coordinate RMSD: 0.0005 Å.

This is retained historical evidence. OpenMM was not rerun on 2026-11-09. The log
contains rounded values; it does not establish the precision of unreported digits.
The original workflow uses local environment setup and is not the fresh-install
entry point for this release.

## Current test status

On 2026-11-09, the publication checkout, including the existing potential and
pressure modules, produced **78 passed, 1 failed** in the pinned environment:

```sh
uv run --frozen python -m pytest tests/test_energy_fun.py \
  tests/test_hessian_eigenvals.py tests/test_pimc_energy.py \
  tests/test_pimc_io.py tests/test_potentials_aziz.py tests/test_pressure.py
```

This selection includes comparison against saved OpenMM values; it does not
execute OpenMM.

The classical minimization and two three-slice PIMC minimization tests also passed
(3 passed; the 100-slice case was not run in this focused check). The standalone
16-slice demonstration passed its convergence and curvature checks separately.

The failing case is `tests/test_pimc_energy.py::test_aziz_to_N64_adrian`:

| Quantity (kB·K) | Computed | Existing reference |
|---|---:|---:|
| Quantum energy estimator | −549.913957 | −618.482609 |
| Interaction energy | −1342.729248 | −1404.928040 |

The test already labels the mismatch as unresolved. Its provenance and potential,
cutoff and estimator conventions require reconciliation. The reference values and
assertions have been preserved. A full-suite pass is not claimed.

## Dependencies

The environment pins JAX/JAXLIB 0.4.38 and the JAX-MD fork at
[`1d768f9`](https://github.com/nbzy1995/jax-md/tree/1d768f9051d552662bd861dd589e71c6bb236c57).
The fork's [cutoff change](https://github.com/nbzy1995/jax-md/commit/1ef73376c17721a78222f9a44eda9902fddcd45c)
uses the quintic switching function `1 − 6x⁵ + 15x⁴ − 10x³`, with continuous first
and second derivatives at both boundaries. Its form follows
[OpenMM's switching-function definition](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#lennard-jones-interaction).

The previous unbounded dependency resolution installed JAX 0.9.0.1, which cannot
import this older fork's `jax.lib.xla_bridge` dependency. The lockfile records the
working dependency set. Compatibility with newer JAX versions is not established.

Existing interactive notebooks can be used with the optional dependencies:

```sh
uv sync --frozen --extra dev --extra demo --extra notebooks
```
