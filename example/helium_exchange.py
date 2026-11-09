"""Reproduce a two-particle exchange-cycle minimization and its figures.

Run from the repository root: uv run python example/helium_exchange.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax_md import space

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from jax_landscape.energy_fun import build_energy_fn_aziz_1995_no_neighborlist
from jax_landscape.hessian_eigenvals import compute_hessian_eigenvalues
from jax_landscape.io.pimc import Path as Worldline
from jax_landscape.io.pimc import load_pimc_worldline_file
from jax_landscape.local_minima import find_local_minimum
from jax_landscape.pimc_energy import build_pimc_energy_fn, build_pimc_energy_fn_xyz


def initial_path(slices=16):
    """Deterministic perturbed ring, closed through a two-particle exchange."""
    rng = np.random.default_rng(7)
    rows = []
    for m in range(slices):
        for n in range(2):
            angle = np.pi * (n + m / slices)
            xyz = np.array([2.2 * np.cos(angle), 1.5 * np.sin(angle), 0.0])
            xyz += rng.normal(0, 0.18, 3)
            prev = ((m - 1) % slices, 1 - n if m == 0 else n)
            nxt = ((m + 1) % slices, 1 - n if m == slices - 1 else n)
            rows.append([m, n, 1, *xyz, *prev, *nxt])
    return Worldline(rows)


def render(output, frames, iterations, energy):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    curve = fig.add_subplot(122)
    colors = ["#286a9e", "#d18b22"]
    lines = [
        ax.plot([], [], [], "o-", color=c, markersize=3, linewidth=1.5)[0]
        for c in colors
    ]
    for axis in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        axis(-2.8, 2.8)
    ax.set(xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-60)
    ax.set_title("Two-particle exchange cycle")
    curve.plot(iterations, energy, color="#286a9e", linewidth=1.8)
    (dot,) = curve.plot([], [], "o", color="#d18b22", markersize=7)
    curve.set(
        xlabel="Optimization iteration",
        ylabel=r"$U_{RP}/k_B$ (K)",
        title="Energy minimization",
    )
    curve.grid(alpha=0.2)
    label = curve.text(0.97, 0.94, "", ha="right", va="top", transform=curve.transAxes)

    def update(i):
        xyz = frames[i]
        for n, line in enumerate(lines):
            # The last bead of each worldline connects to the other particle.
            chain = np.concatenate([xyz[:, n], xyz[:1, 1 - n]])
            line.set_data_3d(chain[:, 0], chain[:, 1], chain[:, 2])
        dot.set_data([iterations[i]], [energy[i]])
        label.set_text(f"Iteration {iterations[i]}\nEnergy {energy[i]:.3f} K")
        return *lines, dot, label

    update(len(frames) - 1)
    fig.savefig(output / "helium-exchange.png", dpi=180)
    # Display recorded optimizer states; no coordinate interpolation.
    selected = np.unique(
        np.linspace(0, len(frames) - 1, min(48, len(frames))).astype(int)
    )
    selected = [*selected, *([len(frames) - 1] * 12)]
    animation = FuncAnimation(fig, update, frames=selected, interval=125)
    animation.save(output / "helium-exchange.gif", writer=PillowWriter(fps=8), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".scratch/helium-exchange"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    path = initial_path()
    displacement, _ = space.free()
    pair = build_energy_fn_aziz_1995_no_neighborlist(displacement)
    ring = build_pimc_energy_fn(displacement, pair)
    energy_fn, template = build_pimc_energy_fn_xyz(
        ring, path, beta=1.0, hbar=3.4812756477, mass=1.0
    )
    energy_fn = jax.jit(energy_fn)
    trajectory = args.output / "trajectory.dat"
    result = find_local_minimum(
        energy_fn,
        jnp.asarray(path.beadCoord),
        gtol=1e-6,
        maxiter=500,
        escape_saddles=True,
        max_saddle_escapes=2,
        trajectory_file=str(trajectory),
        trajectory_path_template=template,
        save_trajectory_every=1,
        log_file=str(args.output / "optimization.csv"),
        log_every=1,
    )
    xyz = result.pop("xyz_final")
    result = {
        key: value.item() if isinstance(value, (jax.Array, np.generic)) else value
        for key, value in result.items()
    }
    grad = jax.grad(lambda x: energy_fn(x)["energy"])(xyz)
    spectrum = compute_hessian_eigenvalues(energy_fn, xyz)["eigenvalues"]
    result.update(
        {
            "gradient_norm": float(jnp.linalg.norm(grad)),
            "negative_eigenvalues": int(jnp.sum(spectrum < -1e-6)),
            "eigenvalue_tolerance": 1e-6,
            "smallest_eigenvalue": float(spectrum[0]),
            "particles": 2,
            "time_slices": 16,
            "temperature_K": 1.0,
            "mass_in_helium_units": 1.0,
            "hbar_reduced": 3.4812756477,
            "seed": 7,
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        }
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if (
        not result["success"]
        or result["gradient_norm"] > 1e-6
        or result["smallest_eigenvalue"] < -1e-6
    ):
        raise RuntimeError(f"Demonstration did not meet the numerical checks: {result}")
    recorded = load_pimc_worldline_file(str(trajectory))
    iterations = np.array(sorted(recorded))
    frames = [recorded[i].beadCoord for i in iterations]
    energies = [float(energy_fn(jnp.asarray(x))["energy"]) for x in frames]
    render(args.output, frames, iterations, energies)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
