import jax.numpy as jnp
from jax_md import smap, space
from jax_landscape.energy_fun import build_energy_fn_aziz_1995_neighborlist, build_energy_fn_aziz_1995_no_neighborlist
import json
import argparse

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_dtype_bits", "64")

# TODO: read command line arguments, including:
# 
# input coordinates file
# box length
# num of particles
# number of beads
# mode: energy, pressure, hessian, or minimization.

# Print the commands with arguments read.
# each command leads to a functionallity of the package


def load_input_data(filename):
    """Loads input data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return {
        'R': jnp.array(data['xyz']),
        'box': jnp.array(data['box']),
        'energy': data['Etot'],
        'grad': jnp.array(data['grad_E'])
    }


def main():

    parser = argparse.ArgumentParser(description='Jax utility for energy landscape analysis.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file containing coordinates and box information.')
    # parser.add_argument('--use_neighbor_list', type=bool, default=True,
                        # help='Use neighbor list for energy calculation.')
    parser.add_argument('--run_minimize', type=bool, default=False,
                        help='Run minimization.')

    args = parser.parse_args()

    print(f"Input file: {args.input_file}")
    # if args.use_neighbor_list:
    #     print("Neighbor list for energy calculation: True")
    # else:
    #     print("Neighbor list for energy calculation: False")
    if args.run_minimize:
        print("Run minimization: True")

    # Load input data
    print("Loading input file...")
    input_data = load_input_data(args.input_file)
    R = input_data['R']
    box_size = input_data['box']
    Etot = input_data['energy']
    print(f"Done. ")

    # setup geometry
    displacement, _ = space.periodic(box_size)

    # Build energy function
    neighbor_fn, energy_fn = build_energy_fn_aziz_1995_neighborlist(displacement, box_size)
    nbrs = neighbor_fn.allocate(R)
    energy = energy_fn(R, neighbor=nbrs)



if __name__ == "__main__":
    main()