"""
Local minimization routines using scipy.optimize.minimize with JAX energy functions.

Usage Examples:

    # Classical without neighbor list
    from jax_md import space
    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    result = find_local_minimum(energy_fn, xyz_initial)

    # Classical with neighbor list
    displacement_fn, _ = space.periodic(box_size)
    neighbor_fn, energy_fn = build_energy_fn_aziz_1995_neighborlist(displacement_fn, box_size)
    result = find_local_minimum(energy_fn, xyz_initial, neighbor_fn=neighbor_fn)

    # PIMC minimization
    from jax_landscape.pimc_energy import make_pimc_minimization_energy_fn
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
    minimization_energy_fn = make_pimc_minimization_energy_fn(
        pimc_energy_fn, path_obj, beta, hbar, mass
    )
    result = find_local_minimum(minimization_energy_fn, path_obj.beadCoord)
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as optimize


def find_local_minimum(
    energy_fn,
    xyz_initial,
    neighbor_fn=None,
    method='L-BFGS-B',
    gtol=1e-8,
    maxiter=50000,
    maxfun=100000,
    log_file=None,
    log_every=10
):
    """
    Find local minimum (inherent structure) of a system using scipy.optimize.minimize.

    Args:
        energy_fn: Pre-built energy function mapping xyz -> scalar energy.
                   If neighbor_fn is provided, should accept (xyz, neighbor=nbrs).
        xyz_initial: Initial coordinates, shape (N, 3) for classical or (M, N, 3) for PIMC
        neighbor_fn: Optional neighbor list allocation function. If provided, neighbor lists
                     will be automatically reallocated during optimization.
        method: Optimization method for scipy.optimize.minimize
        gtol: gradient tolerance for convergence
        maxiter: Maximum number of iterations
        maxfun: Maximum number of function evaluations
        log_file: Optional file path to log optimization details.
        log_every: Log progress every N iterations.

    Returns:
        dict: Optimization results containing minimized coordinates and energy
    """

    # Calculate initial energy
    if neighbor_fn is not None:
        initial_nbrs = neighbor_fn.allocate(xyz_initial)
        initial_energy = energy_fn(xyz_initial, neighbor=initial_nbrs)
    else:
        initial_energy = energy_fn(xyz_initial)

    # Use a closure and a mutable list for the counter to avoid a class.
    iteration_count = [0] 

    if log_file:
        with open(log_file, 'w') as f:
            f.write("Iteration,Energy,GradientNorm\n")

    def objective_function(xyz_flat):
        """The objective is to minimize the total potential energy of the system"""
        iteration_count[0] += 1
        xyz = jnp.array(xyz_flat, dtype=jnp.float64).reshape(xyz_initial.shape)

        if neighbor_fn is not None:
            # Reallocate neighbor list for current coordinates
            nbrs = neighbor_fn.allocate(xyz)
            energy = energy_fn(xyz, neighbor=nbrs)
            grad = jax.grad(energy_fn, argnums=0)(xyz, neighbor=nbrs)
        else:
            energy = energy_fn(xyz)
            grad = jax.grad(energy_fn)(xyz)

        if log_file and iteration_count[0] % log_every == 0:
            grad_norm = np.linalg.norm(grad.flatten())
            with open(log_file, 'a') as f:
                f.write(f"{iteration_count[0]},{energy},{grad_norm}\n")

        return float(energy), np.array(grad.flatten())

    xyz_flat = xyz_initial.flatten()
    
    options = {
        'maxiter': maxiter,
        'maxfun': maxfun, 
        'gtol': gtol
    }
    
    # Run optimization
    results = optimize.minimize(
        objective_function, 
        xyz_flat, 
        method=method, 
        jac=True, 
        options=options
    )
    
    minimized_xyz = results.x.reshape(xyz_initial.shape)
    minimized_energy = results.fun

    return {
        'xyz_final': minimized_xyz,
        'energy_final': minimized_energy,
        'energy_initial': float(initial_energy),
        'success': results.success,
        'message': results.message,
        'nfev': results.nfev,
        'nit': results.nit
    }
