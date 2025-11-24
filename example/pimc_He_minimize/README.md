# PIMC Helium Minimization Examples

This directory contains examples of PIMC (Path Integral Monte Carlo) energy minimization for Helium systems using the Aziz 1995 potential.

## Directory Structure

```
pimc_He_minimize/
├── README.md                        # This file
├── minimize_pimc.py                 # Generic minimization script
├── visualize_minimization.ipynb     # Interactive visualization notebook
├── vis-worldlines.ipynb             # Basic worldline visualization
│
├── N64_cycle1/                      # Example 1: N=64, single cycle
│   ├── input/                       # Input worldline files
│   ├── output/                      # Minimization results
│   └── run.sh                       # Run script
│
├── N64_cycle_large/                 # Example 2: N=64, large cycle
│   ├── input/
│   ├── output/
│   └── run.sh
│
└── N2_cycle2_w1/                    # Example 3: N=2, 2-cycle with winding
    ├── input/
    ├── output/
    └── run.sh
```

## Quick Start

### Running an Example

Each example has a `run.sh` script that executes minimization with appropriate parameters:

```bash
cd N64_cycle1
./run.sh
```

Or for the N=2 example:

```bash
cd N2_cycle2_w1
./run.sh
```

### Visualizing Results

After running minimization, visualize the results using the interactive notebook:

```bash
jupyter notebook visualize_minimization.ipynb
```

In the notebook, change the `example_name` variable to match your example:
```python
example_name = 'N64_cycle1'  # or 'N64_cycle_large', 'N2_cycle2_w1'
```

## Examples

### N64_cycle1: 64-Particle Single Cycle

**System parameters:**
- N = 64 particles
- T = 2.5 K
- Box size = 14.73 Å
- M = 160 time slices
- Cycle structure: Single cycle (no permutation)

**Files:**
- Input: `N64_cycle1/input/N64-cycle1.dat`
- Output: `N64_cycle1/output/`

### N64_cycle_large: 64-Particle Large Cycle

**System parameters:**
- N = 64 particles
- T = 1.55 K  (lower temperature)
- Box size = 14.73 Å
- M = 160 time slices
- Cycle structure: Large cycle with permutation

**Files:**
- Input: `N64_cycle_large/input/N64-cycle_large.dat`
- Output: `N64_cycle_large/output/`

### N2_cycle2_w1: 2-Particle Exchange with Winding

**System parameters:**
- N = 2 particles
- T = 1.0 K
- Box size = 20.0 Å
- M = 100 time slices
- Cycle structure: 2-cycle (particle exchange), winding number 1

**Files:**
- Input: `N2_cycle2_w1/input/N2-Nbeads100-cycle2.dat`
- Output: `N2_cycle2_w1/output/`

## Output Files

After running minimization, each example's `output/` directory contains:

- `minimized.wl.dat`: Final minimized worldline configuration
- `minimized.est.dat`: CSV with initial and final energies for each configuration
- `conf{i}.log`: Energy log for each iteration (CSV with columns: Iteration, Energy(Urp), E_sp, E_int, GradientNorm)
- `conf{i}.trajectory.dat`: Trajectory file with snapshots saved every N iterations

## Generic Minimization Script

The `minimize_pimc.py` script can be used to minimize any PIMC configuration:

```bash
python minimize_pimc.py <input_dir> <output_dir> --N <particles> --box-size <size> --T <temp> [options]
```

**Required arguments:**
- `input_dir`: Directory containing input worldline file (*.dat)
- `output_dir`: Directory for output files
- `--N`: Number of particles
- `--box-size`: Box size in Angstroms
- `--T`: Temperature in Kelvin

**Optional arguments:**
- `--mass`: Particle mass (default: 4.0026 for He-4)
- `--hbar`: Reduced Planck constant (default: 7.638)
- `--configs`: Configuration indices to minimize (default: 0)
- `--save-every`: Save trajectory every N iterations (default: 10)
- `--maxiter`: Maximum iterations (default: 10000)
- `--escape-saddles`: Enable saddle point escape mechanism (default: False)
- `--max-saddle-escapes`: Maximum number of saddle escape attempts (default: 5)

**Example:**
```bash
python minimize_pimc.py \
    N2_cycle2_w1/input \
    N2_cycle2_w1/output \
    --N 2 \
    --box-size 20.0 \
    --T 1.0 \
    --configs 0 \
    --save-every 10
```

**Example with saddle escape:**
```bash
python minimize_pimc.py \
    N2_cycle2_w1/input \
    N2_cycle2_w1/output \
    --N 2 \
    --box-size 20.0 \
    --T 1.0 \
    --escape-saddles \
    --max-saddle-escapes 5
```

## Visualization

### Energy Evolution

The minimization process is tracked through energy components:
- **Urp (total)**: Total ring-polymer energy = E_sp + E_int
- **E_sp (spring)**: Quantum kinetic energy from bead connections
- **E_int (interaction)**: Classical potential energy (Aziz 1995)

During minimization:
- E_sp typically decreases (beads move closer together)
- E_int may increase or decrease (depends on particle arrangements)
- Urp (total) always decreases

### 3D Worldline Visualization

The interactive notebook provides:
- **3D View**: Full 3D visualization of worldline paths
- **2D XY Projection**: Top-down view with 3×3 periodic boundary replication
- **Interactive Slider**: Navigate through minimization trajectory
- **Color Coding**: Beads colored by cycle length
- **Energy Display**: Shows Urp, E_sp, E_int for each snapshot

## Physical Interpretation

### PIMC Worldlines

In PIMC, each particle is represented by M "beads" (imaginary-time slices) connected by harmonic springs:
- **Bead coordinates**: `beadCoord[m, n, :]` = position of particle n at time slice m
- **Connectivity**: `next[m, n]` and `prev[m, n]` define worldline topology
- **Cycles**: In boson systems, worldlines can form permutation cycles

### Energy Components

- **E_sp**: Measures quantum delocalization (smaller = more compact)
- **E_int**: Classical Aziz potential averaged over all beads
- **E_qm**: Quantum correction = E_sp - M/(β²ℏ²) * N * d

### Minimization Goal

Find the local minimum of the PIMC action:
```
Urp = E_sp + E_int
```

This corresponds to an "inherent structure" or instantonic configuration.

## Troubleshooting

**Error: No .dat files found**
- Ensure input directory contains a worldline file

**Error: Trajectory file not found**
- Run the minimization first using `./run.sh` in the example directory

**Minimization not converging**
- Increase `--maxiter`
- Check initial energy - very high values may indicate numerical issues
- Try different configurations using `--configs`

**Visualization shows empty plots**
- Verify output files exist in the example's `output/` directory
- Check that the `example_name` in the notebook matches your example

## Adding New Examples

To add a new example:

1. Create directory structure:
   ```bash
   mkdir -p new_example/input new_example/output
   ```

2. Place input worldline file in `new_example/input/`

3. Create `new_example/run.sh`:
   ```bash
   #!/bin/bash
   cd ..
   python minimize_pimc.py \
       new_example/input \
       new_example/output \
       --N <particles> \
       --box-size <size> \
       --T <temp> \
       --configs 0
   ```

4. Make executable: `chmod +x new_example/run.sh`

5. Run: `cd new_example && ./run.sh`

6. Update `visualize_minimization.ipynb` to add system parameters:
   ```python
   system_params = {
       ...
       'new_example': {'N': <particles>, 'box_size': <size>},
   }
   ```

## References

- Aziz 1995 potential: R. A. Aziz and M. J. Slaman, J. Chem. Phys. 94, 8047 (1991)
- PIMC method: Ceperley, Rev. Mod. Phys. 67, 279 (1995)
