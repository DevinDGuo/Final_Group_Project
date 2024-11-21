#!/bin/bash

set -e

make clean all

rows=(10 15 20 23 30 34 35 61)
cols=(1 2 3 4 5)
processes=(1 2 3 4 5 6 7 8 9 10)

for r in "${rows[@]}"; do
    for c in "${cols[@]}"; do
        for p in "${processes[@]}"; do
            ./make-2d "$r" "$c" A.dat

            mpirun --oversubscribe -np "$p" ./mpi-omp-stencil-2d 100 A.dat B.dat 0
        done
    done
done

echo "Testing Completed"

make clean
