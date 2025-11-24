#!/bin/bash
# Run PIMC minimization for N2_cycle2_w1_true example
#
# System: N=2 Helium atoms, 2-cycle (exchange), TRUE winding number 1

cd ..

python minimize_pimc.py \
    N2_cycle2_w1_true/input \
    N2_cycle2_w1_true/output \
    --N 2 \
    --box-size 20.0 \
    --T 1.0 \
    --mass 4.0026 \
    --hbar 7.638 \
    --configs 0 \
    --save-every 10 \
    --maxiter 5000
