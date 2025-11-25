#!/bin/bash
# Run PIMC minimization for N4_cycle4_w0 example
#
# System: N=4 Helium atoms, 4-cycle (all particles exchange in one cycle)
# Winding number 0: worldline stays within box without wrapping

cd ..

python minimize_pimc.py \
    N4_cycle4_w0/input \
    N4_cycle4_w0/output \
    --N 4 \
    --box-size 20.0 \
    --T 1.0 \
    --mass 4.0026 \
    --hbar 7.638 \
    --configs 0 \
    --save-every 10 \
    --maxiter 5000
