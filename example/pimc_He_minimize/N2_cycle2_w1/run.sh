#!/bin/bash
# Run PIMC minimization for N2_cycle2_w1 example
#
# System: N=2 Helium atoms, 2-cycle (exchange), winding number 1
# The worldline wraps the box once total (each particle crosses half the box)

cd ..

python minimize_pimc.py \
    N2_cycle2_w1/input \
    N2_cycle2_w1/output \
    --N 2 \
    --box-size 20.0 \
    --T 1.0 \
    --mass 4.0026 \
    --hbar 7.638 \
    --configs 0 \
    --save-every 10 \
    --maxiter 5000
