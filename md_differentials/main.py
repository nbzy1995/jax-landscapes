import jax.numpy as jnp
from jax_md import smap, space
from md_differentials.aziz1995 import total_energy_aziz_1995_neighbor_list, total_energy_aziz_1995_no_nl
import json
import argparse

import jax
jax.config.update("jax_enable_x64", True)

# TODO: read command line arguments, including:
# 
# input coordinates file
# box length
# num of particles
# number of beads
# mode: energy, pressure, hessian, or minimization.


# Print the commands with arguments read.


def load_test_data(filename):
    """Loads test data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return {
        'R': jnp.array(data['xyz']),
        'box': jnp.array(data['box']),
        'energy': data['Etot']
    }


def main():
    """
    Main function to calculate the energy of a system.
    """
    parser = argparse.ArgumentParser(description='Differential Utility for Molecular dynamics simulation.')
    parser.add_argument('--input_file', type=str, default='tests/test_data/aziz1995-N6-Nbeads1.json',
                        help='Input file with coordinates and box information.')
    parser.add_argument('--mode', type=str, default='all',
                        help='Calculation mode: energy_no_neighbor_list, energy, or all.')

    args = parser.parse_args()

    print(f"Input file: {args.input_file}")
    print(f"Mode: {args.mode}")

    # Load the input data
    test_data = load_test_data(args.input_file)
    R = test_data['R']
    box_size = test_data['box']
    Etot = test_data['energy']

    displacement, shift = space.periodic(box_size)

    # Calculate the energy using the Aziz 1995 potential
    if args.mode in ['energy_no_neighbor_list', 'all']:
        energy_fn = total_energy_aziz_1995_no_nl(displacement)
        energy = energy_fn(R)
        print(f"Energy (no neighbor list): {energy}")

    if args.mode in ['energy', 'all']:
        neighbor_fn, energy_fn = total_energy_aziz_1995_neighbor_list(displacement, box_size)
        nbrs = neighbor_fn.allocate(R)
        energy = energy_fn(R, neighbor=nbrs)
        print(f"Energy (with neighbor list): {energy}")
    
    print(f"Reference data total energy: {Etot}")

if __name__ == "__main__":
    main()

