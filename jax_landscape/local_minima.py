"""
Local minimization routines using scipy.optimize.minimize with JAX energy functions.

Usage Examples:

    # Classical without neighbor list
    from jax_md import space
    displacement_fn, _ = space.periodic(box_size)
    energy_fn = build_energy_fn_aziz_1995_no_neighborlist(displacement_fn)
    result = find_local_minimum(energy_fn, xyz_initial)

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


def _detect_saddle_point(energy_fn, xyz, threshold, log_file=None):
    """
    Detect if configuration is a saddle point via Hessian eigenanalysis.

    Args:
        energy_fn: Energy function that takes xyz and returns energy (scalar or dict)
        xyz: Configuration coordinates
        threshold: Eigenvalue threshold for saddle detection (e.g., -1e-6)
        log_file: Optional log file path

    Returns:
        (is_saddle, eigenvalues, eigenvectors)
        - is_saddle: True if negative eigenvalues found
        - eigenvalues: Sorted array (ascending)
        - eigenvectors: Corresponding eigenvectors

    Raises:
        Exception if Hessian computation fails (caller should handle)
    """
    from .hessian_eigenvals import compute_hessian_eigenvalues

    # Compute Hessian with eigenvectors
    hessian_result = compute_hessian_eigenvalues(
        energy_fn,
        xyz,
        return_hessian=False,
        return_eigenvectors=True,
        sort_eigenvalues=True
    )

    eigenvalues = hessian_result['eigenvalues']
    eigenvectors = hessian_result['eigenvectors']

    # Count negative eigenvalues
    n_negative = jnp.sum(eigenvalues < threshold)
    min_eigenvalue = jnp.min(eigenvalues)
    is_saddle = n_negative > 0

    # Log analysis
    if log_file:
        with open(log_file, 'a') as f:
            f.write("#\n")
            f.write("# " + "="*66 + "\n")
            f.write("# Hessian Eigenvalue Analysis\n")
            f.write("# " + "="*66 + "\n")
            f.write(f"# Total eigenvalues: {len(eigenvalues)}\n")
            f.write(f"# Negative eigenvalues (λ < {threshold:.0e}): {n_negative}\n")
            f.write(f"# Min eigenvalue: {min_eigenvalue:.6e}\n")

            # Format first 6 eigenvalues nicely
            eigs_str = ", ".join([f"{e:.6e}" for e in eigenvalues[:6]])
            f.write(f"# Smallest 6 eigenvalues: [{eigs_str}]\n")

            f.write(f"# Result: {'SADDLE POINT DETECTED' if is_saddle else 'TRUE LOCAL MINIMUM FOUND'}\n")
            f.write("#\n")

    return is_saddle, eigenvalues, eigenvectors


def _validate_and_escape_saddle(energy_fn, xyz, energy_current, eigenvalues,
                                 eigenvectors, perturbation_magnitudes, log_file=None):
    """
    Validate saddle point and find optimal perturbation along negative eigenvector.

    Strategy:
    1. Try multiple perturbation magnitudes along most negative eigenvector
    2. Check if energy decreases as predicted by eigenvalue
    3. Raise error if energy doesn't decrease (contradiction)
    4. Return perturbation with best energy decrease

    Args:
        energy_fn: Energy function
        xyz: Current configuration at saddle point
        energy_current: Energy at current configuration
        eigenvalues: Sorted eigenvalues (ascending)
        eigenvectors: Corresponding eigenvectors
        perturbation_magnitudes: Tuple of magnitudes to try (e.g., (0.01, 0.05, 0.1, 0.2))
        log_file: Optional log file path

    Returns:
        xyz_perturbed: Best perturbed configuration

    Raises:
        RuntimeError: If energy doesn't decrease along negative eigenvector
    """
    # Most negative eigenvector (index 0, already sorted)
    most_negative_eigenval = eigenvalues[0]
    most_negative_eigenvec = eigenvectors[:, 0]

    # Reshape to xyz shape
    eigenvec_reshaped = most_negative_eigenvec.reshape(xyz.shape)

    # Log escape header
    if log_file:
        with open(log_file, 'a') as f:
            f.write("# " + "="*66 + "\n")
            f.write("# ESCAPING SADDLE POINT\n")
            f.write("# " + "="*66 + "\n")
            f.write(f"# Most negative eigenvalue: {most_negative_eigenval:.6e}\n")
            f.write(f"# Current energy: {energy_current:.8f}\n")
            f.write(f"# Testing perturbation magnitudes: {perturbation_magnitudes}\n")
            f.write("#\n")
            f.write("# Perturbation validation:\n")
            f.write(f"# {'Magnitude':<12} {'Energy':<18} {'ΔE':<18} {'Status'}\n")
            f.write(f"# {'-'*64}\n")

    # Detect if energy function returns dict
    test_result = energy_fn(xyz)
    is_pimc_energy = isinstance(test_result, dict)

    # Try each perturbation magnitude
    best_energy = energy_current
    best_xyz = None
    best_magnitude = None
    found_decrease = False

    for magnitude in perturbation_magnitudes:
        # Perturb along negative eigenvector
        xyz_perturbed = xyz + magnitude * eigenvec_reshaped

        # Compute energy at perturbed configuration
        result = energy_fn(xyz_perturbed)
        energy_perturbed = result['energy'] if is_pimc_energy else result

        delta_E = energy_perturbed - energy_current

        # Check if energy decreased
        energy_decreased = delta_E < -1e-8  # Small tolerance for numerical noise

        if energy_decreased:
            found_decrease = True
            if energy_perturbed < best_energy:
                best_energy = energy_perturbed
                best_xyz = xyz_perturbed
                best_magnitude = magnitude

        # Log result
        status = "DECREASED ✓" if energy_decreased else "INCREASED"
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"# {magnitude:<12.3f} {energy_perturbed:<18.8f} {delta_E:<18.6e} {status}\n")

    # Check if we found any decrease
    if not found_decrease:
        error_msg = (
            f"Saddle point validation failed: Energy did not decrease along negative eigenvector.\n"
            f"Most negative eigenvalue: {most_negative_eigenval:.6e}\n"
            f"This indicates a contradiction - the negative eigenvalue suggests energy should decrease.\n"
            f"Possible causes: numerical error in Hessian, or eigenvalue near zero."
        )
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"#\n# ERROR: {error_msg}\n#\n")
        raise RuntimeError(error_msg)

    # Log selected perturbation
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"#\n")
            f.write(f"# Selected perturbation magnitude: {best_magnitude:.3f}\n")
            f.write(f"# Energy after perturbation: {best_energy:.8f}\n")
            f.write(f"# Energy decrease: {best_energy - energy_current:.8f}\n")
            f.write(f"#\n")

    return best_xyz


def _create_hessp_function(energy_fn, xyz_shape, is_pimc_energy):
    """
    Create Hessian-vector product (hessp) callback for trust-region methods.

    The hessp function computes H*p where H is the Hessian matrix at point x,
    using JAX's forward-mode autodiff (jvp) for efficiency.

    Args:
        energy_fn: Energy function that takes xyz and returns energy (scalar or dict)
        xyz_shape: Shape of the coordinate array (for reshaping)
        is_pimc_energy: Whether energy_fn returns a dict (True) or scalar (False)

    Returns:
        hessp: Function with signature hessp(x_flat, p_flat) -> H*p
    """
    def energy_scalar(xyz_flat):
        """Extract scalar energy from potentially dict-valued energy function."""
        xyz = xyz_flat.reshape(xyz_shape)
        result = energy_fn(xyz)
        return result['energy'] if is_pimc_energy else result

    # Create gradient function
    grad_fn = jax.grad(energy_scalar)

    def hessp(x_flat, p_flat):
        """
        Compute Hessian-vector product H*p at point x.

        Uses the identity: H*p = d/dx[grad(f)] * p = JVP(grad(f), p)
        """
        # Compute JVP: derivative of grad_fn at x in direction p
        _, hvp = jax.jvp(grad_fn, (x_flat,), (p_flat,))
        # Convert to numpy array for scipy
        return np.array(hvp)

    return hessp


def find_local_minimum(
    energy_fn,
    xyz_initial,
    method='trust-ncg',
    gtol=1e-8,
    energy_change_tol=1e-2,
    maxiter=50000,
    maxfun=100000,
    log_file=None,
    log_every=10,
    trajectory_file=None,
    trajectory_path_template=None,
    save_trajectory_every=10,
    initial_iteration=0,
    resume_mode=False,
    metadata=None,
    escape_saddles=True,
    max_saddle_escapes=5,
    saddle_eigenvalue_threshold=-1e-6,
    perturbation_magnitudes=(0.01, 0.05, 0.1, 0.2),
    **kwargs
):
    """
    Find local minimum (inherent structure) of a system using scipy.optimize.minimize.

    WARNING: Do NOT use energy functions with neighbor lists during minimization.
             During optimization, particles can move outside the neighbor list cell,
             causing incorrect energy/gradient calculations. Always use energy functions
             without neighbor lists (e.g., build_energy_fn_aziz_1995_no_neighborlist).

    Args:
        energy_fn: Pre-built energy function mapping xyz -> scalar energy (or dict for PIMC).
                   MUST be built without neighbor lists.
        xyz_initial: Initial coordinates, shape (N, 3) for classical or (M, N, 3) for PIMC
        method: Optimization method for scipy.optimize.minimize (default: 'trust-ncg')
        gtol: Gradient tolerance for convergence (default: 1e-8)
        energy_change_tol: Absolute energy change tolerance for convergence (default 1e-2).
                    Converges when: abs(E_new - E_old) < energy_change_tol AND grad_norm < gtol
        maxiter: Maximum number of iterations (default: 50000)
        maxfun: Maximum number of function evaluations (default: 100000)
        log_file: Optional file path to log optimization details.
        log_every: Log progress every N iterations (default: 10).
        trajectory_file: Optional file path to save minimization trajectory in PIMC worldline format.
                        Only used if trajectory_path_template is also provided.
        trajectory_path_template: Optional Path object providing connectivity structure for PIMC trajectory output.
        save_trajectory_every: Save trajectory snapshot every N iterations (default: 10).
        initial_iteration: Starting iteration number (for resume functionality, default: 0).
        resume_mode: If True, append to existing log/trajectory files instead of overwriting.
        metadata: Optional dict of metadata to write as comments in log file (e.g., system params).
        escape_saddles: If True, automatically detect and escape saddle points using Hessian eigenanalysis.
                       Recommended for trust-region methods. Adds computational cost due to Hessian computation.
                       (default: True).
        max_saddle_escapes: Maximum number of saddle-escape attempts (default: 5).
        saddle_eigenvalue_threshold: Eigenvalues below this are considered negative (default: -1e-6).
        perturbation_magnitudes: Tuple of perturbation sizes to try along negative eigenvector
                                (default: (0.01, 0.05, 0.1, 0.2)).

    Returns:
        dict: Optimization results containing:
            - xyz_final: Final minimized coordinates
            - energy_final: Final energy
            - energy_initial: Initial energy
            - success: Whether optimization succeeded
            - message: Optimization message
            - nfev: Total function evaluations (across all escape attempts)
            - nit: Total iterations (across all escape attempts)
            - saddle_escapes_performed: Number of saddle escapes (0 if disabled)
            - final_is_saddle: Whether final config is saddle (None if escape_saddles=False)
            - final_min_eigenvalue: Minimum Hessian eigenvalue at final config (None if not computed)

    Notes:
        - Saddle escape is computationally expensive (requires full Hessian computation)
        - Recommended for small-medium systems (< 100 degrees of freedom)
        - Trajectory continues across saddle escapes (perturbation jumps are logged)
    """

    # Check if user accidentally passed neighbor_fn (now prohibited)
    if 'neighbor_fn' in kwargs and kwargs['neighbor_fn'] is not None:
        raise ValueError(
            "neighbor_fn is no longer supported in find_local_minimum(). "
            "During minimization, particles can move outside the neighbor list cell, "
            "causing incorrect energy/gradient calculations. "
            "Please use energy functions without neighbor lists "
            "(e.g., build_energy_fn_aziz_1995_no_neighborlist)."
        )

    # Calculate initial energy and detect if PIMC energy function
    initial_result = energy_fn(xyz_initial)

    # Detect if this is a PIMC energy function (returns dict)
    is_pimc_energy = isinstance(initial_result, dict)
    initial_energy = initial_result['energy'] if is_pimc_energy else initial_result

    # Cache shape information for flatten/reshape utilities
    xyz_shape = xyz_initial.shape

    def _energy_scalar_from_flat(x_flat):
        """Scalar energy for flattened coordinates (JAX-friendly)."""
        xyz = x_flat.reshape(xyz_shape)
        result = energy_fn(xyz)
        return result['energy'] if is_pimc_energy else result

    # Pre-define gradient function so we compile it only once.
    grad_fn = jax.grad(_energy_scalar_from_flat)

    def _energy_components_from_flat(x_flat):
        """Return energy and optional components from flattened coordinates."""
        xyz = x_flat.reshape(xyz_shape)
        result = energy_fn(xyz)
        if is_pimc_energy:
            return result['energy'], result['E_sp'], result['E_int']
        return result, None, None

    def _evaluate_state_from_flat(x_flat_np):
        """
        Evaluate energy components and gradient given a numpy-flattened array.

        Returns:
            energy (float-like), grad (jnp.ndarray), E_sp, E_int
        """
        x_flat_jnp = jnp.asarray(x_flat_np)
        energy_val, E_sp_val, E_int_val = _energy_components_from_flat(x_flat_jnp)
        grad_val = grad_fn(x_flat_jnp)
        return energy_val, grad_val, E_sp_val, E_int_val

    # Cumulative tracking across all saddle-escape attempts
    cumulative_iterations = initial_iteration
    cumulative_nfev = 0

    # Variables for saddle-escape loop
    is_saddle = None
    eigenvals = None
    eigenvecs = None

    # Custom exception for convergence
    class ConvergenceReached(Exception):
        pass

    # Setup log file (once, before loop)
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
                f.write(f"#   Custom convergence criteria (scipy criteria disabled):\n")
                f.write(f"#     abs(delta_E) < {energy_change_tol} AND grad_norm < {gtol}\n")
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

    # Setup trajectory file with header (once, before loop)
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

    # BEGIN SADDLE-ESCAPE LOOP
    for escape_attempt in range(max_saddle_escapes + 1):
        if log_file and escape_attempt > 0:
            with open(log_file, 'a') as f:
                f.write("#\n")
                f.write("# " + "="*66 + "\n")
                f.write(f"# SADDLE ESCAPE ATTEMPT {escape_attempt}\n")
                f.write("# " + "="*66 + "\n")
                f.write("#\n")
        # --- Per-attempt state -------------------------------------------------
        attempt_start_step = cumulative_iterations

        start_x_flat = np.asarray(xyz_initial).reshape(-1)
        start_energy_val, start_E_sp_val, start_E_int_val = _energy_components_from_flat(jnp.asarray(start_x_flat))
        start_energy = float(np.asarray(start_energy_val))
        start_E_sp = float(np.asarray(start_E_sp_val)) if is_pimc_energy and start_E_sp_val is not None else None
        start_E_int = float(np.asarray(start_E_int_val)) if is_pimc_energy and start_E_int_val is not None else None

        iterations_completed = [attempt_start_step]
        last_logged_step = [attempt_start_step]
        last_snapshot_step = [attempt_start_step]
        raw_eval_counter = [0]

        current_step = {
            'energy': start_energy,
            'grad_norm': 0.0,
            'xyz': np.asarray(xyz_initial),
            'E_sp': start_E_sp,
            'E_int': start_E_int,
            'converged': False,
            'convergence_message': None
        }

        # Scratch space for the most recent evaluation returned by SciPy.
        pending_eval = {
            'x': None,
            'energy': None,
            'grad': None,
            'grad_norm': None,
            'E_sp': None,
            'E_int': None,
            'xyz': None
        }

        def refresh_pending_eval(x_flat_np):
            energy_val, grad_val, E_sp_val, E_int_val = _evaluate_state_from_flat(x_flat_np)
            grad_np = np.asarray(grad_val)
            grad_np = grad_np.astype(float, copy=False)
            pending_eval['x'] = np.asarray(x_flat_np, dtype=float).copy()
            pending_eval['energy'] = float(np.asarray(energy_val))
            pending_eval['grad'] = grad_np
            pending_eval['grad_norm'] = float(np.linalg.norm(grad_np))
            if is_pimc_energy:
                pending_eval['E_sp'] = float(np.asarray(E_sp_val))
                pending_eval['E_int'] = float(np.asarray(E_int_val))
            else:
                pending_eval['E_sp'] = None
                pending_eval['E_int'] = None
            pending_eval['xyz'] = pending_eval['x'].reshape(xyz_shape).copy()
            return pending_eval

        # --- Logging helpers ---------------------------------------------------
        def log_iteration(step_idx, energy_val, grad_norm_val, E_sp_val=None, E_int_val=None, force=False):
            if not log_file:
                return
            if not force:
                if log_every <= 0 or (step_idx - initial_iteration) % log_every != 0:
                    return
            with open(log_file, 'a') as f:
                if is_pimc_energy:
                    f.write(f"{step_idx},{energy_val},{E_sp_val},{E_int_val},{grad_norm_val}\n")
                else:
                    f.write(f"{step_idx},{energy_val},{grad_norm_val}\n")
            last_logged_step[0] = step_idx

        def save_snapshot(step_idx, xyz_coords, force=False):
            if trajectory_handle is None:
                return
            if not force:
                if save_trajectory_every <= 0 or (step_idx - initial_iteration) % save_trajectory_every != 0:
                    return
            path_snapshot = create_path_snapshot(xyz_coords)
            write_pimc_worldline_config(trajectory_handle, path_snapshot, step_idx)
            trajectory_handle.flush()
            last_snapshot_step[0] = step_idx

        # --- Functions passed to SciPy -----------------------------------------
        def objective_function(x_flat):
            raw_eval_counter[0] += 1
            x_flat_np = np.asarray(x_flat)
            refresh_pending_eval(x_flat_np)
            return pending_eval['energy'], pending_eval['grad']

        def step_callback(xk):
            """Executed by SciPy each time it executes a new iterate step."""
            xk_np = np.asarray(xk)
            if pending_eval['x'] is None or not np.allclose(pending_eval['x'], xk_np):
                refresh_pending_eval(xk_np)

            iteration_id = iterations_completed[0] + 1
            iterations_completed[0] = iteration_id

            prev_energy = current_step['energy']

            current_step['energy'] = pending_eval['energy']
            current_step['grad_norm'] = pending_eval['grad_norm']
            current_step['xyz'] = pending_eval['xyz']
            if is_pimc_energy:
                current_step['E_sp'] = pending_eval['E_sp']
                current_step['E_int'] = pending_eval['E_int']

            log_iteration(
                iteration_id,
                current_step['energy'],
                current_step['grad_norm'],
                current_step.get('E_sp'),
                current_step.get('E_int')
            )

            save_snapshot(iteration_id, current_step['xyz'])

            energy_change = abs(current_step['energy'] - prev_energy)
            if energy_change < energy_change_tol and current_step['grad_norm'] < gtol:
                current_step['converged'] = True
                current_step['convergence_message'] = (
                    f"CUSTOM CONVERGENCE: Energy change {energy_change:.6e} < {energy_change_tol:.6e} "
                    f"AND gradient norm {current_step['grad_norm']:.6e} < {gtol:.6e}"
                )
                raise ConvergenceReached()

        x0_flat = np.asarray(xyz_initial).reshape(-1)

        # Detect trust-region methods that require Hessian information
        trust_region_methods = ['trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']
        uses_trust_region = method in trust_region_methods

        options = {
            'maxiter': maxiter,
            'disp': True      # Enable verbose output
        }

        # For L-BFGS-B: disable all convergence criteria, rely on custom callback
        if method == 'L-BFGS-B':
            options['maxfun'] = maxfun
            # ftol: stop when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
            # Setting to 0 effectively disables (controlled by factr = ftol / eps)
            options['ftol'] = 0
            # gtol: stop when max{|proj g_i|} <= gtol
            # Setting to very small value (1e-100) makes this criterion almost impossible to satisfy
            options['gtol'] = 1e-100

        # For trust-region methods: use scipy defaults, but disable gtol to rely on custom callback
        elif uses_trust_region:
            # Trust-region methods use gtol differently; set to very small to rely on callback
            options['gtol'] = 1e-100

        # Create hessp (Hessian-vector product) for trust-region methods
        hessp_callback = None
        if uses_trust_region and method != 'trust-exact':
            # trust-exact computes full Hessian internally, others use hessp
            hessp_callback = _create_hessp_function(energy_fn, xyz_initial.shape, is_pimc_energy)

        # Run optimization with convergence callback
        minimize_kwargs = {
            'fun': objective_function,
            'x0': x0_flat,
            'method': method,
            'jac': True,
            'options': options,
            'callback': step_callback
        }

        if hessp_callback is not None:
            minimize_kwargs['hessp'] = hessp_callback

        # --- Execute SciPy minimize -------------------------------------------
        custom_result_used = False

        try:
            results = optimize.minimize(**minimize_kwargs)
        except ConvergenceReached:
            custom_result_used = True

            class CustomResult:
                def __init__(self):
                    self.x = current_step['xyz'].reshape(-1)
                    self.fun = current_step['energy']
                    self.success = True
                    self.message = current_step['convergence_message']
                    self.nfev = raw_eval_counter[0]
                    self.nit = iterations_completed[0] - attempt_start_step

            results = CustomResult()

        attempt_iterations = iterations_completed[0] - attempt_start_step
        attempt_evals = raw_eval_counter[0]

        if iterations_completed[0] > last_logged_step[0]:
            log_iteration(
                iterations_completed[0],
                current_step['energy'],
                current_step['grad_norm'],
                current_step.get('E_sp'),
                current_step.get('E_int'),
                force=True
            )

        if trajectory_handle and iterations_completed[0] > last_snapshot_step[0]:
            save_snapshot(iterations_completed[0], current_step['xyz'], force=True)

        if not custom_result_used:
            results.nfev = attempt_evals
            results.nit = attempt_iterations
            results.fun = current_step['energy']
            results.x = current_step['xyz'].reshape(-1)

        minimized_xyz = results.x.reshape(xyz_shape)
        minimized_energy = results.fun

        cumulative_nfev += attempt_evals
        cumulative_iterations = iterations_completed[0]

        if log_file:
            with open(log_file, 'a') as f:
                f.write("#\n")
                f.write("# Optimization completed\n")
                f.write(f"# Success: {results.success}\n")
                f.write(f"# Message: {results.message}\n")
                f.write(f"# Completed iterations (nit): {results.nit}\n")
                f.write(f"# Function evaluations (nfev): {results.nfev}\n")
                f.write(f"# Final step index: {iterations_completed[0]}\n")
                f.write(f"# Energy at this attempt: {minimized_energy:.6f}\n")

        # If saddle escape disabled, stop here
        if not escape_saddles:
            break

        # Try to detect saddle point
        try:
            is_saddle, eigenvals, eigenvecs = _detect_saddle_point(
                energy_fn, minimized_xyz, saddle_eigenvalue_threshold, log_file
            )
        except Exception as e:
            # Hessian computation failed - log and stop
            if log_file:
                with open(log_file, 'a') as f:
                    f.write("#\n")
                    f.write("# " + "="*66 + "\n")
                    f.write("# HESSIAN COMPUTATION FAILED\n")
                    f.write("# " + "="*66 + "\n")
                    f.write(f"# Error: {str(e)}\n")
                    f.write("# Stopping saddle-escape procedure.\n")
                    f.write("# Recording current configuration as minimum.\n")
                    f.write("#\n")
            is_saddle = False
            eigenvals = None
            eigenvecs = None
            break

        # If true minimum found, stop
        if not is_saddle:
            break

        # If exhausted attempts, stop with warning
        if escape_attempt >= max_saddle_escapes:
            if log_file:
                with open(log_file, 'a') as f:
                    f.write("#\n")
                    f.write("# WARNING: Maximum saddle escapes reached\n")
                    f.write("# Final configuration is still a saddle point\n")
                    f.write("#\n")
            break

        # Validate and escape saddle point
        xyz_initial = _validate_and_escape_saddle(
            energy_fn, minimized_xyz, minimized_energy,
            eigenvals, eigenvecs, perturbation_magnitudes, log_file
        )

        # Log perturbation to trajectory if requested
        if trajectory_handle:
            path_snapshot = create_path_snapshot(xyz_initial)
            write_pimc_worldline_config(trajectory_handle, path_snapshot, cumulative_iterations)
            trajectory_handle.flush()

        # Loop continues with new xyz_initial

    # END SADDLE-ESCAPE LOOP

    # Write final summary to log file
    if log_file:
        with open(log_file, 'a') as f:
            f.write("# " + "="*66 + "\n")
            f.write("# FINAL SUMMARY\n")
            f.write("# " + "="*66 + "\n")

            if escape_saddles:
                f.write(f"# Saddle escape enabled: Yes\n")
                f.write(f"# Total saddle escapes performed: {escape_attempt}\n")

                if is_saddle:
                    f.write(f"# Final configuration: SADDLE POINT (max escapes reached)\n")
                    f.write(f"# WARNING: Did not converge to true local minimum\n")
                else:
                    f.write(f"# Final configuration: True local minimum\n")

                if eigenvals is not None:
                    f.write(f"# Final min eigenvalue: {eigenvals[0]:.6e}\n")
            else:
                f.write(f"# Saddle escape enabled: No\n")

            f.write(f"# Total iterations (all attempts): {cumulative_iterations}\n")
            f.write(f"# Total function evaluations: {cumulative_nfev}\n")
            f.write(f"# Final energy: {minimized_energy:.8f}\n")
            f.write(f"# Initial energy: {initial_energy:.8f}\n")
            f.write(f"# Total energy reduction: {initial_energy - minimized_energy:.8f}\n")
            f.write("#\n")

    # Close trajectory file
    if trajectory_handle:
        trajectory_handle.close()

    return {
        'xyz_final': minimized_xyz,
        'energy_final': minimized_energy,
        'energy_initial': float(initial_energy),
        'success': results.success,
        'message': results.message,
        'nfev': cumulative_nfev,
        'nit': cumulative_iterations - initial_iteration,
        'saddle_escapes_performed': escape_attempt,
        'final_is_saddle': is_saddle if escape_saddles else None,
        'final_min_eigenvalue': float(eigenvals[0]) if eigenvals is not None else None
    }
