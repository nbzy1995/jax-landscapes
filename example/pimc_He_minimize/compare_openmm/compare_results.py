#!/usr/bin/env python3
"""
Compare minimization results between OpenMM and JAX implementations.
"""
import sys
import numpy as np
import pandas as pd

print("=" * 70)
print("OpenMM vs JAX Minimization Comparison")
print("=" * 70)
print()

# Parameters
kjmol_to_kbk = 120.2722922542  # 1 kJ/mol = 120.2722922542 kB·K
M = 100  # Number of beads
relative_tol = 1e-3  # 0.1% tolerance

print(f"Note: OpenMM sums over {M} beads, JAX averages → divide OpenMM by {M}")
print(f"Tolerance: {relative_tol*100:.1f}% relative error")
print()

# Load OpenMM results
print("Loading OpenMM results...")
try:
    openmm_df = pd.read_csv('openmm_output/min_energy.csv')
    openmm_E0 = float(openmm_df['E_0'].values[0])  # kJ/mol
    openmm_Esp0 = float(openmm_df['Esp_0'].values[0])  # kJ/mol
    openmm_Emin = float(openmm_df['E_min'].values[0])  # kJ/mol
    openmm_Esp_min = float(openmm_df['Esp_min'].values[0])  # kJ/mol

    # Convert to kB·K and adjust for M factor
    openmm_E0_kbk = openmm_E0 * kjmol_to_kbk / M
    openmm_Esp0_kbk = openmm_Esp0 * kjmol_to_kbk / M
    openmm_Eint0_kbk = (openmm_E0 - openmm_Esp0) * kjmol_to_kbk / M
    openmm_Emin_kbk = openmm_Emin * kjmol_to_kbk / M
    openmm_Esp_min_kbk = openmm_Esp_min * kjmol_to_kbk / M
    openmm_Eint_min_kbk = (openmm_Emin - openmm_Esp_min) * kjmol_to_kbk / M

    print(f"  ✓ Loaded from openmm_output/min_energy.csv")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Load JAX results
print("Loading JAX results...")
try:
    jax_df = pd.read_csv('jax_output/conf0.log', comment='#')

    # Initial (first row) and final (last row)
    jax_E0 = float(jax_df.iloc[0]['Energy(Urp)'])
    jax_Esp0 = float(jax_df.iloc[0]['E_sp'])
    jax_Eint0 = float(jax_df.iloc[0]['E_int'])
    jax_Emin = float(jax_df.iloc[-1]['Energy(Urp)'])
    jax_Esp_min = float(jax_df.iloc[-1]['E_sp'])
    jax_Eint_min = float(jax_df.iloc[-1]['E_int'])

    print(f"  ✓ Loaded from jax_output/conf0.log")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print()

# Energy comparison
print("=" * 70)
print("Energy Comparison (all in kB·K)")
print("=" * 70)
print()

def compare(name, openmm_val, jax_val, tol):
    """Compare two values and return pass/fail."""
    diff = abs(openmm_val - jax_val)
    rel_err = diff / abs(jax_val) if abs(jax_val) > 1e-10 else 0.0
    passed = rel_err < tol
    status = "✓" if passed else "✗"

    print(f"{name:30s}  OpenMM: {openmm_val:12.2f}  JAX: {jax_val:12.2f}  "
          f"Diff: {diff:10.2f}  ({rel_err*100:6.3f}%)  {status}")
    return passed

# Compare all energy components
print("INITIAL ENERGIES:")
init_E_pass = compare("Total Energy", openmm_E0_kbk, jax_E0, relative_tol)
init_Esp_pass = compare("Spring Energy (E_sp)", openmm_Esp0_kbk, jax_Esp0, relative_tol)
init_Eint_pass = compare("Interaction Energy (E_int)", openmm_Eint0_kbk, jax_Eint0, relative_tol)
print()

print("FINAL ENERGIES:")
final_E_pass = compare("Total Energy", openmm_Emin_kbk, jax_Emin, relative_tol)
final_Esp_pass = compare("Spring Energy (E_sp)", openmm_Esp_min_kbk, jax_Esp_min, relative_tol)
final_Eint_pass = compare("Interaction Energy (E_int)", openmm_Eint_min_kbk, jax_Eint_min, relative_tol)
print()

# Coordinate comparison
print("=" * 70)
print("Coordinate Comparison")
print("=" * 70)
print()

coord_pass = False
try:
    openmm_coords = np.loadtxt('openmm_output/minimized.pos', skiprows=1)

    sys.path.insert(0, '/Users/Yang/Documents/WorkSpace/jax-landscape')
    from jax_landscape.io.pimc import load_pimc_worldline_file
    jax_paths = load_pimc_worldline_file('jax_output/conf0.trajectory.dat')
    jax_path = jax_paths[max(jax_paths.keys())]

    # Reshape JAX to match OpenMM format
    N = 64
    jax_coords = np.zeros((M * N, 3))
    for n in range(N):
        for m in range(M):
            jax_coords[n * M + m] = jax_path.beadCoord[m, n]

    rmsd = np.sqrt(np.mean((jax_coords - openmm_coords)**2))
    print(f"RMSD: {rmsd:.4f} Angstrom")

    if rmsd < 0.01:
        print("✓ PASS: Identical minima (RMSD < 0.01 Å)")
        coord_pass = True
    elif rmsd < 0.1:
        print("✓ PASS: Equivalent minima (RMSD < 0.1 Å)")
        coord_pass = True
    else:
        print("✗ FAIL: Different minima (RMSD > 0.1 Å)")

except Exception as e:
    print(f"✗ Cannot compare coordinates: {e}")

print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

all_energy_pass = (init_E_pass and init_Esp_pass and init_Eint_pass and
                   final_E_pass and final_Esp_pass and final_Eint_pass)
all_pass = all_energy_pass and coord_pass

print(f"Initial energies:  {'✓ PASS' if (init_E_pass and init_Esp_pass and init_Eint_pass) else '✗ FAIL'}")
print(f"Final energies:    {'✓ PASS' if (final_E_pass and final_Esp_pass and final_Eint_pass) else '✗ FAIL'}")
print(f"Coordinates:       {'✓ PASS' if coord_pass else '✗ FAIL'}")
print()

if all_pass:
    print("✓ VALIDATION PASSED")
else:
    print("✗ VALIDATION FAILED")

print()
sys.exit(0 if all_pass else 1)
