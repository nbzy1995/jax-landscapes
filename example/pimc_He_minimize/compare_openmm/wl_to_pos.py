#!/usr/bin/env python3
"""
Convert JAX worldline file (.wl.dat) to OpenMM position file (.pos).
"""
import sys
import numpy as np

sys.path.insert(0, '/Users/Yang/Documents/WorkSpace/jax-landscape')
from jax_landscape.io.pimc import load_pimc_worldline_file


def convert_wl_to_pos(wl_file, pos_file, box_size):
    """Convert worldline file to .pos format.

    Args:
        wl_file: Path to worldline file (.wl.dat)
        pos_file: Output .pos file path
        box_size: Box size in Angstrom (assumed cubic)
    """
    print(f"Converting {wl_file} -> {pos_file}")

    # Load worldline
    paths = load_pimc_worldline_file(wl_file, Lx=box_size, Ly=box_size, Lz=box_size)
    path = paths[list(paths.keys())[0]]  # Get first (or only) configuration

    M = path.numTimeSlices
    N = path.numParticles

    print(f"  Time slices (M): {M}")
    print(f"  Particles (N): {N}")
    print(f"  Total beads: {M * N}")

    # Convert to .pos ordering: particle 0 all beads, particle 1 all beads, etc.
    # beadCoord shape: (M, N, 3) -> need (M*N, 3) with proper ordering
    coords = np.zeros((M * N, 3))
    for n in range(N):
        for m in range(M):
            idx = n * M + m
            coords[idx] = path.beadCoord[m, n]

    # Write .pos file
    with open(pos_file, 'w') as f:
        # Header: 0 box_x box_y box_z
        f.write(f"0 {box_size} {box_size} {box_size}\n")
        # Coordinates
        for coord in coords:
            f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

    print(f"  ✓ Written to {pos_file}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python wl_to_pos.py <input.wl.dat> <output.pos> <box_size>")
        sys.exit(1)

    wl_file = sys.argv[1]
    pos_file = sys.argv[2]
    box_size = float(sys.argv[3])

    convert_wl_to_pos(wl_file, pos_file, box_size)
