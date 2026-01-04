#!/bin/bash

################################################################################
# Compare JAX-landscape vs OpenMM local minimization
################################################################################
#
# Goal: Verify that JAX and OpenMM minimization find the same energy minimum
#
# Setup: 64 He atoms, 1-cycle PIMC config, starting near a minimum
#
# Test: Both methods minimize from the same perturbed configuration
#       and should converge to identical final coordinates and energies.
#
# Usage:
#   bash run.sh
#
# Input required:
#   - N64-cycle1.conf0.wl.dat (worldline format config file)
#   - The script will automatically convert to .pos format for OpenMM
#
# Output:
#   - openmm_output/: OpenMM minimized results
#   - jax_output/: JAX minimized results
#   - Comparison printed to stdout
#
################################################################################

set -e  # Exit on error

# Configuration
BOX_SIZE=14.321030  # Angstrom

echo "================================================================================"
echo "JAX-landscape vs OpenMM Minimization Comparison"
echo "================================================================================"

# Step 0: Convert worldline to .pos format for OpenMM
echo ""
echo "Step 0: Converting worldline to .pos format..."
source ../../../.venv/bin/activate
python wl_to_pos.py N64-cycle1.conf0.wl.dat N64-cycle1.conf0.pos $BOX_SIZE

# Step 1: Verify worldline format conversion
echo ""
echo "Step 1: Verifying worldline file format..."
python verify_worldline_conversion.py

# Step 2: Run OpenMM minimization
echo ""
echo "Step 2: Running OpenMM minimization..."
eval "$(conda shell.bash hook)"
conda activate openmm_8.1.0

python minimize_pimd_He_openmm8.py \
  N64.pdb \
  in_pars.txt \
  N64-cycle1.conf0.pos \
  openmm_output/minimized.pos \
  openmm_output/min_energy.csv

# Step 3: Run JAX minimization
echo ""
echo "Step 3: Running JAX minimization..."
conda deactivate
source ../../../.venv/bin/activate

python ../minimize_pimc.py N64-cycle1.conf0.wl.dat jax_output \
    --N 64 --box-size $BOX_SIZE --T 2.5 \
    --mass 1.0 --hbar 3.4812756477 \
    --configs 0 --save-every 100 --maxiter 50000

# Step 4: Compare results
echo ""
echo "Step 4: Comparing results..."
python compare_results.py

echo ""
echo "================================================================================"
echo "Done!"
echo "================================================================================"
