"""
Hessian matrix computation and eigenvalue analysis using JAX autodiff.

Usage Examples:

    # Classical energy function
    from jax_md import space
    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    result = compute_hessian_eigenvalues(energy_fn, xyz)

    # PIMC energy function
    from jax_landscape.pimc_energy import build_pimc_energy_fn
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
    from jax_landscape.pimc_energy import build_pimc_energy_fn_xyz
    minimization_energy_fn, _ = build_pimc_energy_fn_xyz(
        pimc_energy_fn, path_obj, beta, hbar, mass
    )
    result = compute_hessian_eigenvalues(minimization_energy_fn, path_obj.beadCoord)

    # Request full hessian matrix (memory intensive for large systems)
    result = compute_hessian_eigenvalues(energy_fn, xyz, return_hessian=True)
"""

import jax
import jax.numpy as jnp


def compute_hessian_eigenvalues(
    energy_fn,
    xyz,
    return_hessian=False,
    return_eigenvectors=True,
    sort_eigenvalues=True
):
    """
    Compute Hessian matrix and its eigenvalues for a given configuration.

    Args:
        energy_fn: Pre-built energy function mapping xyz -> scalar energy.
                   Works with classical (N, 3) or PIMC (M, N, 3) configurations.
        xyz: Configuration coordinates, shape (N, 3) for classical or (M, N, 3) for PIMC
        return_hessian: If True, return full Hessian matrix (default False, saves memory)
        return_eigenvectors: If True, return eigenvectors (default True)
        sort_eigenvalues: If True, sort eigenvalues in ascending order (default True)

    Returns:
        dict containing:
            - eigenvalues: Array of eigenvalues, shape (D,) where D = N*3 or M*N*3
            - eigenvectors: Array of eigenvectors, shape (D, D) [if return_eigenvectors=True]
            - hessian: Full Hessian matrix, shape (D, D) [if return_hessian=True]
            - energy: Scalar energy at the given configuration
            - xyz_shape: Original shape of xyz for reference
    """

    original_shape = xyz.shape
    xyz_flat = xyz.flatten()

    # Define energy function for flattened coordinates
    def energy_flat(xyz_1d):
        xyz_reshaped = xyz_1d.reshape(original_shape)
        result = energy_fn(xyz_reshaped)
        # Handle both scalar and dict return values (PIMC energy functions)
        if isinstance(result, dict):
            return result['energy']
        return result

    # Compute Hessian using JAX autodiff
    hessian_fn = jax.hessian(energy_flat)
    hessian = hessian_fn(xyz_flat)

    # Compute eigenvalues and eigenvectors using symmetric solver
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)

    # Sort if requested
    if sort_eigenvalues:
        sort_idx = jnp.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

    # Compute energy at this configuration
    energy_val = energy_flat(xyz_flat)

    # Build result dictionary
    result = {
        'eigenvalues': eigenvalues,
        'energy': float(energy_val),
        'xyz_shape': original_shape
    }

    if return_eigenvectors:
        result['eigenvectors'] = eigenvectors

    if return_hessian:
        result['hessian'] = hessian

    return result
