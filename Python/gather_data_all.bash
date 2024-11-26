#!/bin/bash

# Usage: ./gather_data.sh <pthread_csv_file> <omp_csv_file> <mpi_csv_file> <mpi_omp_csv_file>

# Check for correct number of arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <pthread_csv_file> <omp_csv_file> <mpi_csv_file> <mpi_omp_csv_file>"
    exit 1
fi

# Output CSV files
PTHREAD_CSV="../data/$1"
OMP_CSV="../data/$2"
MPI_CSV="../data/$3"
MPI_OMP_CSV="../data/$4"

# Create data directory if it doesn't exist
mkdir -p "../data"

# Headers for CSV files
echo "Matrix Size,Threads,Time Overall (s),Time Computation (s),Time Other (s)" > "$PTHREAD_CSV"
echo "Matrix Size,Threads,Time Overall (s),Time Computation (s),Time Other (s)" > "$OMP_CSV"
echo "Matrix Size,Processes,Time Overall (s),Time Computation (s),Time Other (s)" > "$MPI_CSV"
echo "Matrix Size,Processes,OMP Threads,Time Overall (s),Time Computation (s),Time Other (s)" > "$MPI_OMP_CSV"

# Matrix sizes, thread counts, and process counts
MATRIX_SIZES=(50 100 150 200 250)
THREAD_COUNTS=(1 2 4)
PROCESS_COUNTS=(1 2 4)
OMP_THREAD_COUNTS=(1 2 4)
TS=12

# Input and output files
INPUT_FILE="initial.dat"
OUTPUT_FILE="final.dat"

# Change to 'code' directory
cd ../code || exit

# Clean and compile the project
make clean all

# Loop over matrix sizes
for n in "${MATRIX_SIZES[@]}"; do
    # Create input data file
    ./make-2d "$n" "$n" "$INPUT_FILE"

    # Run pthread stencil
    for t in "${THREAD_COUNTS[@]}"; do
        echo "Running PTHREAD driver: $TS iterations for matrix size ${n}x${n} with $t threads..."
        OUTPUT=$(./pth-stencil-2d "$TS" "$INPUT_FILE" "$OUTPUT_FILE" 0 "$t")
        TIME_OVERALL=$(echo "$OUTPUT" | grep "Time Overall" | awk '{print $3}')
        TIME_COMP=$(echo "$OUTPUT" | grep "Time Computation" | awk '{print $3}')
        TIME_OTHER=$(echo "$OUTPUT" | grep "Time Other" | awk '{print $3}')
        echo "$n,$t,$TIME_OVERALL,$TIME_COMP,$TIME_OTHER" >> "$PTHREAD_CSV"
    done

    # Run OpenMP stencil
    for t in "${THREAD_COUNTS[@]}"; do
        echo "Running OMP driver: $TS iterations for matrix size ${n}x${n} with $t threads..."
        OUTPUT=$(./omp-stencil-2d "$TS" "$INPUT_FILE" "$OUTPUT_FILE" 0 "$t")
        TIME_OVERALL=$(echo "$OUTPUT" | grep "Time Overall" | awk '{print $3}')
        TIME_COMP=$(echo "$OUTPUT" | grep "Time Computation" | awk '{print $3}')
        TIME_OTHER=$(echo "$OUTPUT" | grep "Time Other" | awk '{print $3}')
        echo "$n,$t,$TIME_OVERALL,$TIME_COMP,$TIME_OTHER" >> "$OMP_CSV"
    done

    # Run MPI stencil
    for p in "${PROCESS_COUNTS[@]}"; do
        echo "Running MPI driver: $TS iterations for matrix size ${n}x${n} with $p processes..."
        OUTPUT=$(mpirun -np "$p" ./mpi-stencil-2d "$TS" "$INPUT_FILE" "$OUTPUT_FILE" 0)
        TIME_OVERALL=$(echo "$OUTPUT" | grep "Time Overall" | awk '{print $3}')
        TIME_COMP=$(echo "$OUTPUT" | grep "Time Computation" | awk '{print $3}')
        TIME_OTHER=$(echo "$OUTPUT" | grep "Time Other" | awk '{print $3}')
        echo "$n,$p,$TIME_OVERALL,$TIME_COMP,$TIME_OTHER" >> "$MPI_CSV"
    done

    # Run MPI-OMP stencil
    for p in "${PROCESS_COUNTS[@]}"; do
        for t in "${OMP_THREAD_COUNTS[@]}"; do
            echo "Running MPI-OMP driver: $TS iterations for matrix size ${n}x${n} with $p processes and $t threads..."
            OUTPUT=$(mpirun -np "$p" ./mpi-omp-stencil-2d "$TS" "$INPUT_FILE" "$OUTPUT_FILE" "$t" 0)
            TIME_OVERALL=$(echo "$OUTPUT" | grep "Time Overall" | awk '{print $3}')
            TIME_COMP=$(echo "$OUTPUT" | grep "Time Computation" | awk '{print $3}')
            TIME_OTHER=$(echo "$OUTPUT" | grep "Time Other" | awk '{print $3}')
            echo "$n,$p,$t,$TIME_OVERALL,$TIME_COMP,$TIME_OTHER" >> "$MPI_OMP_CSV"
        done
    done
done

# Clean up after execution
make clean

echo "Experiments completed. Results saved to $PTHREAD_CSV, $OMP_CSV, $MPI_CSV, and $MPI_OMP_CSV."
