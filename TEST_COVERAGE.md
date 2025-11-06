# Test Coverage

## Overview by Module

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
- PIMC: N=2, M=3, two-cycle configurations
- PIMC: N=512. See `example/pimc_minimize`
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
- [TODO] Large systems: N=500 (memory/performance)
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

