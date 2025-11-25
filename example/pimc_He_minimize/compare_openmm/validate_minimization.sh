#!/bin/bash

# Master validation script for comparing JAX and legacy OpenMM minimization
# This script runs the full validation pipeline

set -e  # Exit on error


# Step 1: Verify worldline conversion
source ../../../.venv/bin/activate
python verify_worldline_conversion.py


# Step 2: Run legacy OpenMM minimization

eval "$(conda shell.bash hook)"

conda activate openmm_8.1.0

python minimize_pimd_He_openmm8.py \
  N64.pdb \
  in_pars.txt \
  N64-cycle1.conf0.pos \
  openmm_output/minimized.pos \
  openmm_output/min_energy.csv



# Step 3: Run new JAX minimization

conda deactivate
source ../../../.venv/bin/activate

python ../minimize_pimc.py . jax_output \
    --N 64 --box-size 14.321030 --T 2.5 \
    --mass 4.0026 --hbar 3.4812756477 \
    --configs 0 --save-every 100 --maxiter 50000 --escape-saddle

# Step 4: Compare results

python compare_results.py