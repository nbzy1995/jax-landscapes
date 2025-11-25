#!/usr/bin/env python3
"""
Verify that the existing worldline file N64-cycle1.conf0.wl.dat is correctly converted
from the .pos file with proper connectivity for distinguishable particles.
"""
import sys
import numpy as np

# Add parent directory to path to import jax_landscape
sys.path.insert(0, '/Users/Yang/Documents/WorkSpace/jax-landscape')

from jax_landscape.io.pimc import load_pimc_worldline_file

def verify_worldline():
    print("=" * 70)
    print("Verifying Worldline File Conversion")
    print("=" * 70)

    # Parameters
    M = 100  # time slices (beads per particle)
    N = 64   # particles
    box_size = 14.321030  # Angstrom

    print(f"\nExpected configuration:")
    print(f"  Time slices (M): {M}")
    print(f"  Particles (N): {N}")
    print(f"  Total beads: {M * N}")
    print(f"  Box size: {box_size} Angstrom")

    # Load worldline file
    print(f"\nLoading worldline file: N64-cycle1.conf0.wl.dat")
    try:
        paths = load_pimc_worldline_file(
            'N64-cycle1.conf0.wl.dat',
            Lx=box_size, Ly=box_size, Lz=box_size
        )
        path = paths[0]
        print(f"  ✓ Loaded successfully")
        print(f"  - Number of time slices: {path.numTimeSlices}")
        print(f"  - Number of particles: {path.numParticles}")
        print(f"  - Total beads: {path.numTimeSlices * path.numParticles}")
    except Exception as e:
        print(f"  ✗ Failed to load worldline file: {e}")
        return False

    # Check dimensions
    print(f"\nChecking dimensions...")
    if path.numTimeSlices != M:
        print(f"  ✗ Wrong number of time slices: {path.numTimeSlices} (expected {M})")
        return False
    if path.numParticles != N:
        print(f"  ✗ Wrong number of particles: {path.numParticles} (expected {N})")
        return False
    print(f"  ✓ Dimensions correct")

    # Load .pos file for coordinate comparison
    print(f"\nLoading .pos file: N64-cycle1.conf0.pos")
    try:
        pos_data = np.loadtxt('N64-cycle1.conf0.pos', skiprows=1)  # angstroms
        print(f"  ✓ Loaded {pos_data.shape[0]} coordinates")
    except Exception as e:
        print(f"  ✗ Failed to load .pos file: {e}")
        return False

    if pos_data.shape[0] != M * N:
        print(f"  ✗ Wrong number of coordinates: {pos_data.shape[0]} (expected {M * N})")
        return False

    # Verify connectivity for distinguishable particles
    print(f"\nVerifying connectivity (distinguishable particles - no permutations)...")
    connectivity_ok = True
    bad_count = 0

    for m in range(M):
        for n in range(N):
            next_m, next_n = path.next[m, n]
            prev_m, prev_n = path.prev[m, n]

            expected_next_m = (m + 1) % M
            expected_prev_m = (m - 1) % M

            if next_m != expected_next_m or next_n != n:
                if bad_count < 5:  # Only print first 5 errors
                    print(f"  ✗ Bad next at (m={m}, n={n}): got ({next_m}, {next_n}), expected ({expected_next_m}, {n})")
                connectivity_ok = False
                bad_count += 1

            if prev_m != expected_prev_m or prev_n != n:
                if bad_count < 5:
                    print(f"  ✗ Bad prev at (m={m}, n={n}): got ({prev_m}, {prev_n}), expected ({expected_prev_m}, {n})")
                connectivity_ok = False
                bad_count += 1

    if not connectivity_ok:
        print(f"  ✗ Found {bad_count} connectivity errors")
        print(f"  This worldline file has particle permutations - need distinguishable particles!")
        return False

    print(f"  ✓ All connectivity correct (distinguishable particles)")

    # Verify cycle structure
    print(f"\nVerifying cycle structure...")
    if not path.is_closed_worldline:
        print(f"  ✗ Worldline is not closed")
        return False

    print(f"  - Number of cycles: {len(path.cycleSizeDist)}")
    print(f"  - Cycle sizes: {path.cycleSizeDist[:10]}{'...' if len(path.cycleSizeDist) > 10 else ''}")

    # Each particle should form its own cycle of length M
    expected_cycle_size = M
    if not np.all(path.cycleSizeDist == expected_cycle_size):
        print(f"  ✗ Not all cycles have size {expected_cycle_size}")
        print(f"  Unique cycle sizes: {np.unique(path.cycleSizeDist)}")
        return False

    if len(path.cycleSizeDist) != N:
        print(f"  ✗ Wrong number of cycles: {len(path.cycleSizeDist)} (expected {N})")
        return False

    print(f"  ✓ All {N} particles form separate cycles of length {M}")

    # Verify coordinates match
    print(f"\nVerifying coordinates match .pos file...")

    # The .pos file format is: particle 0 all beads, particle 1 all beads, etc.
    # Need to reshape appropriately for comparison
    # .pos: [p0b0, p0b1, ..., p0b99, p1b0, p1b1, ..., p1b99, ...]
    # worldline beadCoord shape: (M=100, N=64, 3)

    # Reshape worldline coords to match .pos ordering
    wl_coords_flat = np.zeros((M * N, 3))
    for n in range(N):
        for m in range(M):
            idx = n * M + m  # .pos file index
            wl_coords_flat[idx] = path.beadCoord[m, n]

    coord_diff = np.abs(wl_coords_flat - pos_data)
    max_diff = np.max(coord_diff)
    mean_diff = np.mean(coord_diff)

    print(f"  - Max coordinate difference: {max_diff:.6e} Angstrom")
    print(f"  - Mean coordinate difference: {mean_diff:.6e} Angstrom")

    if not np.allclose(wl_coords_flat, pos_data, atol=1e-6):
        print(f"  ✗ Coordinates do not match (tolerance 1e-6)")
        # Show a few examples of mismatches
        mismatches = np.where(coord_diff > 1e-6)
        if len(mismatches[0]) > 0:
            print(f"  First few mismatches:")
            for i in range(min(5, len(mismatches[0]))):
                idx = mismatches[0][i]
                coord_idx = mismatches[1][i]
                print(f"    Bead {idx}, coord {coord_idx}: wl={wl_coords_flat[idx, coord_idx]:.6f}, pos={pos_data[idx, coord_idx]:.6f}")
        return False

    print(f"  ✓ Coordinates match within tolerance")

    # Final summary
    print("\n" + "=" * 70)
    print("VERIFICATION PASSED")
    print("=" * 70)
    print("The worldline file is correctly formatted for distinguishable particles.")
    print("Ready to use for minimization comparison.")
    print()

    return True

if __name__ == '__main__':
    success = verify_worldline()
    sys.exit(0 if success else 1)
