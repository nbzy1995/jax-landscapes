"""
Aziz 1995 potential implementation for Helium interactions using JAX-MD.
"""

# TODO: make all documentation nicer

import jax.numpy as jnp
from jax_md import energy, space, smap, partition



# -----------
# The two-body potential function for the Aziz 1995 potential
AZIZ_PARAMS = {
    'A_star': 1.86924404e5,
    'alpha_star': 10.5717543,
    'beta_star': -2.07758779,
    'c6': 1.35186623,
    'c8': 0.41495143,
    'c10': 0.17151143,
    'D': 1.438,
    'epsilon': 0.0910933,  # kJ/mol
    'rm': 0.29683,  # nm
}

def aziz_1995(r, **kwargs):
    """
    Aziz 1995 potential for Helium two-body interactions, with cutoff.
    """
    # TODO: will a different way of passing parameters improve performance?
    A_star = kwargs.get('A_star', AZIZ_PARAMS['A_star'])
    alpha_star = kwargs.get('alpha_star', AZIZ_PARAMS['alpha_star'])
    beta_star = kwargs.get('beta_star', AZIZ_PARAMS['beta_star'])
    c6 = kwargs.get('c6', AZIZ_PARAMS['c6'])
    c8 = kwargs.get('c8', AZIZ_PARAMS['c8'])
    c10 = kwargs.get('c10', AZIZ_PARAMS['c10'])
    D = kwargs.get('D', AZIZ_PARAMS['D'])
    epsilon = kwargs.get('epsilon', AZIZ_PARAMS['epsilon'])
    rm = kwargs.get('rm', AZIZ_PARAMS['rm'])

    r_reduced = r / rm

    # Damping function
    Fx = jnp.where(r_reduced < D,
                   jnp.exp(-(D / r_reduced - 1.0)**2),
                   1.0)

    # Repulsive term
    repulsive = A_star * jnp.exp(-alpha_star * r_reduced + beta_star * r_reduced**2)

    # Dispersion term
    r_reduced_inv = 1.0 / r_reduced
    dispersion = (c6 * r_reduced_inv**6 + 
                  c8 * r_reduced_inv**8 + 
                  c10 * r_reduced_inv**10) * Fx

    return epsilon * (repulsive - dispersion)


# -----------
# factory for energy function without neighbor list
def build_energy_fn_aziz_1995_no_neighborlist(
    displacement_or_metric,
    r_cutoff=1.36, 
    r_sw=1.36*0.9,
    **kwargs): 

    r_cutoff = jnp.array(r_cutoff) 
    r_sw = jnp.array(r_sw) # switching distance

    energy_fn = smap.pair(
        energy.multiplicative_isotropic_cutoff(aziz_1995, r_sw, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric)
    )
    return energy_fn


# factory for energy function with neighbor list
def build_energy_fn_aziz_1995_neighborlist(
    displacement_or_metric,
    box_size,
    r_cutoff=1.36, 
    r_sw=1.36*0.9,  # switching distance
    dr_threshold=0.5,  # buffer size for neighbor list
    format=partition.OrderedSparse,
    **kwargs): 

    r_cutoff = jnp.array(r_cutoff, jnp.float64) 
    r_sw = jnp.array(r_sw, jnp.float64) 
    dr_threshold = jnp.array(dr_threshold, jnp.float64)

    neighbor_fn = partition.neighbor_list(
            displacement_or_metric, 
            box_size, 
            r_cutoff, 
            dr_threshold=dr_threshold,
            format=format)

    energy_fn = smap.pair_neighbor_list(
        energy.multiplicative_isotropic_cutoff(aziz_1995, r_sw, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric)
        )

    return neighbor_fn, energy_fn