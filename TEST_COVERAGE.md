# Test Coverage

This document provides a detailed overview of test coverage, methodology, and known gaps.

## Coverage Matrix

| Module | Coverage | Test Cases | Rigor |
|--------|----------|------------|-------|
| `energy_fun.py` | ✓✓ Comprehensive | 2 parametrized (N=6, N=500) | Property-based + numerical validation |
| `pimc_energy.py` | ✓✓ Comprehensive | 4 cases (M=1, cycles, N=64 system) | Mathematical validation + reference data |
| `local_minima.py` | ✓✓ Comprehensive | 3 parametrized (classical, PIMC variants) | 4-property validation + saddle escape |
| `hessian_eigenvals.py` | ✓✓ Comprehensive | 6 tests (analytical, numerical, reconstruction) | Multi-method consistency |
| `io/pimc.py` | ✓ Basic | 1 test (loading only) | Structural validation |
| CLI (`main.py`) | ⚠️ Untested | 0 tests | - |

**Legend:**
- ✓✓ = Rigorous testing with multiple validation methods
- ✓ = Basic functional coverage
- ⚠️ = No test coverage

## Module Details

### Energy Functions (`test_energy_fun.py`)

**What's tested:**
- Energy and gradient accuracy against reference data
- Both neighbor list and no-neighbor-list implementations
- Small (N=6) and large (N=500) systems
- Periodic boundary conditions

**Testing methodology:**
- Reference data from validated implementations (kJ/mol, nm units)
- Relative error < 1e-8 for energy
- Component-wise gradient comparison with allclose
- Consistency between neighbor/no-neighbor methods

**Test cases:**
- `test_energy_grad[N6]`: 6 particles
- `test_energy_grad[N500]`: 500 particles

**Known gaps:**
- No tests for cutoff radius edge cases
- No validation of neighbor list updates during dynamics
- No tests with non-cubic boxes

### PIMC Energy (`test_pimc_energy.py`)

**What's tested:**
- Classical limit (M=1): Spring energy = 0, potential matches direct evaluation
- Free particle (V=0): Spring energy analytical validation
- Cycle configurations: One-cycle and two-cycle worldlines
- Full Aziz potential: N=64 system comparison with Adrian PIMC code

**Testing methodology:**
- Analytical solutions for free particles (spring energy formula)
- Reference data from production PIMC simulations
- Relative error < 1e-6 for N=64 system

**Test cases:**
- `test_classical_M1_matches_slice_energy`: Single-slice configuration
- `test_free_particle_one_cycle`: 2 particles, 3 beads, 1 cycle
- `test_free_particle_two_cycle`: 2 particles, 3 beads, 2 cycles
- `test_aziz_to_N64_adrian`: 64 particles, production PIMC configuration

**Known gaps:**
- No tests with different masses or hbar values
- No validation of temperature dependence

### Local Minimization (`test_local_minima.py`)

**What's tested:**
- 4 mathematical properties of local minima:
  1. Zero gradient (||∇E|| < tolerance)
  2. Energy decrease (E_final < E_initial)
  3. Local stability (random perturbations increase energy)
  4. Hessian positive semi-definite (no negative eigenvalues)
- Saddle point detection and escape
- Trajectory file I/O for PIMC configurations
- Both classical and PIMC minimization

**Testing methodology:**
- Property-based validation (mathematical guarantees)
- Statistical perturbation tests (20 samples × 3 magnitudes)
- Eigenvalue analysis with refined thresholds:
  - Negative: λ < -1e-6 (saddle point)
  - Near-zero: |λ| < 1e-6 (symmetry modes)
- Reference data comparison (relaxed tolerances for path-dependent optimization)

**Test cases:**
- `test_local_minimum_classical_no_neighborlist[N6]`: Classical system
- `test_local_minimum_pimc[cycle1]`: PIMC 2 particles, 1 cycle
- `test_local_minimum_pimc[cycle2]`: PIMC 2 particles, 2 cycles

**Validation details:**
- Gradient tolerance: 1e-6 (trust-ncg typical convergence)
- Perturbation success: ≥99% must increase energy
- Expected symmetry modes: 3 (translation) for general systems, 5 for N=2 PIMC

**Known gaps:**
- No tests with failed optimizations
- No tests for maximum iteration limits
- No validation of log file contents
- No tests with different optimizers (only trust-ncg tested)

### Hessian Analysis (`test_hessian_eigenvals.py`)

**What's tested:**
- Hessian symmetry
- Eigenvalue sorting
- Numerical consistency (autodiff vs finite differences)
- Eigenvalue/eigenvector reconstruction (H·v = λ·v)
- Analytical test cases (harmonic oscillator)
- Both classical and PIMC energy functions

**Testing methodology:**
- Multi-method validation (autodiff, numerical, analytical)
- Central difference scheme for numerical Hessian (4th order)
- Eigenvalue equation verification
- Known analytical solutions (harmonic oscillator: 5 zero modes, 1 vibration)

**Test cases:**
- `test_hessian_classical_small_system`: N=6 particles
- `test_hessian_without_eigenvectors`: Optional return values
- `test_hessian_verification_simple_harmonic`: Analytical validation
- `test_hessian_pimc_energy`: PIMC configuration
- `test_numerical_gradient_consistency`: Autodiff vs finite diff
- `test_eigenvalue_eigenvector_reconstruction`: H·v = λ·v

**Known gaps:**
- No performance tests for large systems (memory scaling)
- No tests for ill-conditioned Hessians
- No validation of eigenvector orthogonality

### PIMC I/O (`test_pimc_io.py`)

**What's tested:**
- File loading and parsing
- Path object construction
- Connectivity arrays (next/prev)
- Cycle detection and size distribution
- Closed worldline validation

**Testing methodology:**
- Structural validation of loaded data
- Connectivity chain traversal
- Cycle length verification

**Test cases:**
- `test_load_sample_worldline`: N=2, M=3 configuration

**Known gaps:**
- ⚠️ **No tests for writing worldline files**
- No tests for malformed input files
- No tests for large configurations (N>64)
- No validation of box size handling
- No tests for edge cases (N=1, M=1)

## Untested Functionality

### CLI Interface (`main.py`)
**Status:** ⚠️ No test coverage

**Functionality:**
- Command-line argument parsing
- Input format selection (classical, PIMC)
- Configuration index selection
- Minimization workflow integration

**Risk:** User-facing interface has no automated validation

### Performance & Scalability
**Status:** ⚠️ No benchmark tests

**Missing validation:**
- Timing for different system sizes
- Memory usage profiling
- JIT compilation overhead
- Neighbor list performance comparison

### Error Handling
**Status:** ⚠️ Minimal coverage

**Missing tests:**
- Invalid input files
- Numerical instabilities (e.g., very small beta)
- Cutoff radius violations
- Convergence failures

## Testing Standards

### Numerical Tolerance
- Energy comparisons: `rtol=1e-8` (float64 precision)
- Gradient comparisons: `rtol=1e-8`, component-wise
- Hessian comparisons: `rtol=1e-6, atol=1e-8` (second derivatives)
- Optimization convergence: `gtol=1e-6` (typical trust-ncg)

### Property-Based Tests
Local minimum validation enforces:
1. ||∇E|| < 1e-6
2. E_final < E_initial
3. ≥99% of perturbations increase energy
4. All eigenvalues > -1e-6 (allowing near-zero symmetry modes)

### Reference Data
- Energy test data: JSON format (nm, kJ/mol → Å, kB·K conversion)
- PIMC test data: Adrian PIMC code output format
- Validation systems: N=6 (small), N=64 (production), N=500 (large)

## Recommendations for Future Testing

### High Priority
1. **CLI integration tests** - Validate end-to-end workflows
2. **PIMC file writing** - Test trajectory export accuracy
3. **Error handling** - Test failure modes and error messages

### Medium Priority
4. **Performance benchmarks** - Automated timing regression tests
5. **Large system validation** - N>500 particles
6. **Alternative optimizers** - Test L-BFGS, CG, etc.

### Low Priority
7. **Non-cubic boxes** - Rectangular periodic boundaries
8. **Edge cases** - N=1, M=1, extreme temperatures
9. **Documentation tests** - Verify examples in docstrings
