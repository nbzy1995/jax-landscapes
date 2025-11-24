#!/bin/bash
# Run PIMC minimization for N2_cycle1_w0 example
#
# System: N=2 Helium atoms, cycle 1 (no particle exchange), winding number 0

cd ..

python ../minimize_pimc.py \
    input \
    output \
    --N 2 \
    --box-size 20.0 \
    --T 1.0 \
    --mass 4.0026 \
    --hbar 7.638 \
    --configs 0 \
    --save-every 10 \
    --maxiter 5000