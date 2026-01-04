#!/bin/bash
# Run PIMC minimization for N64_cycle_large example
#
# System: N=64 Helium atoms, large cycle (with permutation)
# Parameters from test_full_wl in tests/test_pimc_energy.py

cd ..

python minimize_pimc.py \
    N64_cycle_large/input \
    N64_cycle_large/output \
    --N 64 \
    --box-size 14.73 \
    --T 1.55 \
    --mass 1.0 \
    --hbar 3.481 \
    --configs 0 \
    --save-every 10 \
    --maxiter 100000
