#!/bin/bash
# Run PIMC minimization for N2_cycle2_w0 example
#
# System: N=2 Helium atoms, 2-cycle (particle exchange), winding number 0
# This script now loops over all *.dat worldline files in input/ and
# writes results for each one into the shared output/ directory with
# filenames tagged by the input worldline basename.

cd ..

INPUT_DIR="N2_cycle2_w0/input"
OUTPUT_DIR="N2_cycle2_w0/output"

mkdir -p "${OUTPUT_DIR}"

for WL_FILE in "${INPUT_DIR}"/*.dat; do
    # Skip if the glob didn't match anything
    [ -e "${WL_FILE}" ] || continue

    BASENAME=$(basename "${WL_FILE}")
    echo "==============================================="
    echo "Running minimization for worldline file: ${WL_FILE}"
    echo "  Output directory: ${OUTPUT_DIR}"
    echo "==============================================="

    python minimize_pimc.py \
        "${WL_FILE}" \
        "${OUTPUT_DIR}" \
        --N 2 \
        --box-size 20.0 \
        --T 1.0 \
        --mass 4.0026 \
        --hbar 7.638 \
        --configs 0 \
        --save-every 10 \
        --maxiter 5000
done
