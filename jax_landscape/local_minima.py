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

    # PIMC minimization with trajectory saving
    from jax_landscape.pimc_energy import build_pimc_energy_fn_xyz
    pimc_energy_fn = build_pimc_energy_fn(displacement_fn, classical_energy_fn)
    minimization_energy_fn, path_template = build_pimc_energy_fn_xyz(
        pimc_energy_fn, path_obj, beta, hbar, mass
    )
    result = find_local_minimum(
        minimization_energy_fn,
        path_obj.beadCoord,
        trajectory_file='output.dat',
        trajectory_path_template=path_template,
        save_trajectory_every=10
    )
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
    log_every=10,
    trajectory_file=None,
    trajectory_path_template=None,
    save_trajectory_every=10,
    initial_iteration=0,
    resume_mode=False,
    metadata=None
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
        trajectory_file: Optional file path to save minimization trajectory in PIMC worldline format.
                        Only used if trajectory_path_template is also provided.
        trajectory_path_template: Optional Path object providing connectivity structure for PIMC trajectory output.
        save_trajectory_every: Save trajectory snapshot every N iterations (default 10).
        initial_iteration: Starting iteration number (for resume functionality, default 0).
        resume_mode: If True, append to existing log/trajectory files instead of overwriting.
        metadata: Optional dict of metadata to write as comments in log file (e.g., system params).

    Returns:
        dict: Optimization results containing minimized coordinates and energy
    """

    # Calculate initial energy and detect if PIMC energy function
    if neighbor_fn is not None:
        initial_nbrs = neighbor_fn.allocate(xyz_initial)
        initial_result = energy_fn(xyz_initial, neighbor=initial_nbrs)
    else:
        initial_result = energy_fn(xyz_initial)

    # Detect if this is a PIMC energy function (returns dict)
    is_pimc_energy = isinstance(initial_result, dict)
    initial_energy = initial_result['energy'] if is_pimc_energy else initial_result

    # Use a closure and a mutable list for the counter to avoid a class.
    iteration_count = [initial_iteration]

    # Setup log file
    if log_file:
        file_mode = 'a' if resume_mode else 'w'
        with open(log_file, file_mode) as f:
            if not resume_mode:
                # Write metadata as comments
                if metadata:
                    f.write("# Minimization run metadata\n")
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")

                # Write optimization settings
                f.write(f"# Optimization settings:\n")
                f.write(f"#   method: {method}\n")
                f.write(f"#   gtol: {gtol}\n")
                f.write(f"#   maxiter: {maxiter}\n")
                f.write(f"#   maxfun: {maxfun}\n")
                f.write("#\n")

                # Write CSV header
                if is_pimc_energy:
                    f.write("Iteration,Energy(Urp),E_sp,E_int,GradientNorm\n")
                else:
                    f.write("Iteration,Energy,GradientNorm\n")

                # Write iteration 0 (initial state)
                if is_pimc_energy:
                    E_sp = initial_result['E_sp']
                    E_int = initial_result['E_int']
                    f.write(f"0,{initial_energy},{E_sp},{E_int},0.0\n")
                else:
                    f.write(f"0,{initial_energy},0.0\n")
            else:
                # Resume mode - write resume info
                f.write(f"# Resuming from iteration {initial_iteration}\n")

    # Setup trajectory file with header
    trajectory_handle = None
    if trajectory_file and trajectory_path_template:
        file_mode = 'a' if resume_mode else 'w'
        trajectory_handle = open(trajectory_file, file_mode)

        # Import here to avoid circular dependency
        from .io.pimc import write_pimc_worldline_config

        # Define PathSnapshot class for trajectory snapshots
        class PathSnapshot:
            def __init__(self, beadCoord, next, prev, wlIndex, write_order):
                self.beadCoord = beadCoord
                self.next = next
                self.prev = prev
                self.wlIndex = wlIndex
                self.write_order = write_order

        if not resume_mode:
            trajectory_handle.write("# PIMCID: minimization-trajectory\n")

            # Write iteration 0 (initial state)
            initial_snapshot = PathSnapshot(
                beadCoord=np.array(xyz_initial),
                next=trajectory_path_template.next,
                prev=trajectory_path_template.prev,
                wlIndex=trajectory_path_template.wlIndex if hasattr(trajectory_path_template, 'wlIndex') else None,
                write_order=trajectory_path_template.write_order if hasattr(trajectory_path_template, 'write_order') else None
            )
            write_pimc_worldline_config(trajectory_handle, initial_snapshot, 0)
            trajectory_handle.flush()

        # Helper to create Path object from xyz (reuse PathSnapshot class)
        def create_path_snapshot(xyz_coords):
            """Create a Path-like object with current coords and template connectivity."""
            return PathSnapshot(
                beadCoord=np.array(xyz_coords),
                next=trajectory_path_template.next,
                prev=trajectory_path_template.prev,
                wlIndex=trajectory_path_template.wlIndex if hasattr(trajectory_path_template, 'wlIndex') else None,
                write_order=trajectory_path_template.write_order if hasattr(trajectory_path_template, 'write_order') else None
            )

    def objective_function(xyz_flat):
        """The objective is to minimize the total potential energy of the system"""
        iteration_count[0] += 1
        xyz = jnp.array(xyz_flat, dtype=jnp.float64).reshape(xyz_initial.shape)

        if neighbor_fn is not None:
            # Reallocate neighbor list for current coordinates
            nbrs = neighbor_fn.allocate(xyz)
            result = energy_fn(xyz, neighbor=nbrs)
            grad = jax.grad(lambda x: energy_fn(x, neighbor=nbrs) if not is_pimc_energy else energy_fn(x, neighbor=nbrs)['energy'], argnums=0)(xyz)
        else:
            result = energy_fn(xyz)
            grad = jax.grad(lambda x: energy_fn(x) if not is_pimc_energy else energy_fn(x)['energy'])(xyz)

        # Extract scalar energy from result (handle both scalar and dict)
        if is_pimc_energy:
            energy = result['energy']
            E_sp = result['E_sp']
            E_int = result['E_int']
        else:
            energy = result

        if log_file and iteration_count[0] % log_every == 0:
            grad_norm = np.linalg.norm(grad.flatten())
            with open(log_file, 'a') as f:
                if is_pimc_energy:
                    f.write(f"{iteration_count[0]},{energy},{E_sp},{E_int},{grad_norm}\n")
                else:
                    f.write(f"{iteration_count[0]},{energy},{grad_norm}\n")

        # Save trajectory snapshot if requested
        if trajectory_handle and iteration_count[0] % save_trajectory_every == 0:
            path_snapshot = create_path_snapshot(xyz)
            write_pimc_worldline_config(trajectory_handle, path_snapshot, iteration_count[0])
            trajectory_handle.flush()  # Ensure data is written

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

    # Save final trajectory state and close file
    if trajectory_handle:
        path_snapshot = create_path_snapshot(minimized_xyz)
        write_pimc_worldline_config(trajectory_handle, path_snapshot, iteration_count[0])
        trajectory_handle.close()

    return {
        'xyz_final': minimized_xyz,
        'energy_final': minimized_energy,
        'energy_initial': float(initial_energy),
        'success': results.success,
        'message': results.message,
        'nfev': results.nfev,
        'nit': results.nit
    }
