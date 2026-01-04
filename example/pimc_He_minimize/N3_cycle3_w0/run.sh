#!/bin/bash
# Run PIMC minimization for N3_cycle3_w0 example
#
# System: N=3 Helium atoms, 3-cycle (all particles exchange in one cycle)
# Winding number 0: worldline stays within box without wrapping
# This script loops over all *.dat worldline files in input/ and writes
# results for each one into the shared output/ directory.

cd ..

INPUT_DIR="N3_cycle3_w0/input"
OUTPUT_DIR="N3_cycle3_w0/output"

mkdir -p "${OUTPUT_DIR}"

for WL_FILE in "${INPUT_DIR}"/*.dat; do
    [ -e "${WL_FILE}" ] || continue

    echo "==============================================="
    echo "Running minimization for worldline file: ${WL_FILE}"
    echo "  Output directory: ${OUTPUT_DIR}"
    echo "==============================================="

    python minimize_pimc.py \
        "${WL_FILE}" \
        "${OUTPUT_DIR}" \
        --N 3 \
        --box-size 20.0 \
        --T 1.0 \
        --mass 4.0026 \
        --hbar 7.638 \
        --configs 0 \
        --save-every 10 \
        --maxiter 5000
done
