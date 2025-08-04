"""
Local minimization routines using scipy.optimize.minimize with JAX energy functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as optimize
from jax_md import space

from .energy_fun import build_energy_fn_aziz_1995_neighborlist, build_energy_fn_aziz_1995_no_neighborlist


def find_local_minimum(
    energy_fn_factory, 
    xyz_initial,
    box_size,
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
        energy_fn_factory: factory function of the energy function of the system
        xyz_initial: Initial coordinates, shape (N, 3)
        box_size: Box size for periodic boundary conditions
        method: Optimization method for scipy.optimize.minimize
        gtol: gradient tolerance for convergence
        maxiter: Maximum number of iterations
        maxfun: Maximum number of function evaluations
        log_file: Optional file path to log optimization details.
        log_every: Log progress every N iterations.
        
    Returns:
        dict: Optimization results containing minimized coordinates and energy
    """
    
    displacement_fn, shift_fn = space.periodic(box_size)
    
    neighbor_fn = None
    if energy_fn_factory == build_energy_fn_aziz_1995_neighborlist:
        neighbor_fn, energy_fn = energy_fn_factory(displacement_fn, box_size)
    elif energy_fn_factory == build_energy_fn_aziz_1995_no_neighborlist:
        energy_fn = energy_fn_factory(displacement_fn)
    else:
        raise ValueError("Unsupported energy function factory provided.")

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
        xyz = jnp.array(xyz_flat, dtype=jnp.float64).reshape(-1, 3)
        
        if neighbor_fn is not None:
            # Reallocate neighbor list for current coordinates
            nbrs = neighbor_fn.allocate(xyz)
            energy = energy_fn(xyz, neighbor=nbrs)
            grad = jax.grad(energy_fn, argnums=0)(xyz, neighbor=nbrs)
        else:
            energy = energy_fn(xyz)
            grad = jax.grad(energy_fn, argnums=0)(xyz)
        
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
    
    minimized_xyz = results.x.reshape(-1, 3)
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
