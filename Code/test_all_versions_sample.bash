#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Clean and build
make clean all

# Test parameters
sizes=(10 50 100)
iterations=(1 5 10)
mpi_procs=(2 4 8)
omp_threads=(2 4 8)

total_tests=0
failed_tests=0

for size in "${sizes[@]}"; do
    echo "------------------------------------------------------------------------------------"
    echo "TESTING MATRIX SIZE: ${size}x${size}"
    
    # Create input matrix
    ./make-2d $size $size "input_${size}.dat"
    
    for iter in "${iterations[@]}"; do
        echo "------------------------------------------------------------------------------------"
        echo "Running $iter iterations"
        
        # Run sequential version (baseline)
        ./stencil-2d $iter "input_${size}.dat" "seq_${size}_${iter}.out" "all_seq.dat" > /dev/null
        
        # Run all parallel versions and compare outputs
        for threads in "${omp_threads[@]}"; do
            # Run pthread and OpenMP versions
            ./pth-stencil-2d $iter "input_${size}.dat" "pth_${size}_${iter}_${threads}.out" 0 $threads > /dev/null
            ./omp-stencil-2d $iter "input_${size}.dat" "omp_${size}_${iter}_${threads}.out" 0 $threads > /dev/null
            
            echo "DIFFING THREADED IMPLEMENTATION (threads=$threads)"
            if diff3 "seq_${size}_${iter}.out" "pth_${size}_${iter}_${threads}.out" "omp_${size}_${iter}_${threads}.out" > /dev/null 2>&1; then
                echo -e "\t${GREEN}All thread-based implementations match.${NC}"
                ((total_tests++))
            else
                if diff "seq_${size}_${iter}.out" "pth_${size}_${iter}_${threads}.out" > /dev/null 2>&1; then
                    echo -e "\t${YELLOW}OpenMP implementation differs (threads=$threads).${NC}"
                elif diff "seq_${size}_${iter}.out" "omp_${size}_${iter}_${threads}.out" > /dev/null 2>&1; then
                    echo -e "\t${YELLOW}Pthread implementation differs (threads=$threads).${NC}"
                elif diff "omp_${size}_${iter}_${threads}.out" "pth_${size}_${iter}_${threads}.out" > /dev/null 2>&1; then
                    echo -e "\t${YELLOW}Sequential implementation differs from parallel versions.${NC}"
                else
                    echo -e "\t${RED}All implementations produce different results (threads=$threads).${NC}"
                fi
                ((failed_tests++))
                ((total_tests++))
            fi
        done
        
        # Run and compare MPI versions
        for procs in "${mpi_procs[@]}"; do
            mpirun --oversubscribe -np $procs ./mpi-stencil-2d $iter "input_${size}.dat" "mpi_${size}_${iter}_${procs}.out" 0 > /dev/null
            
            echo "DIFFING MPI IMPLEMENTATION (processes=$procs)"
            if diff "seq_${size}_${iter}.out" "mpi_${size}_${iter}_${procs}.out" > /dev/null 2>&1; then
                echo -e "\t${GREEN}MPI implementation matches sequential.${NC}"
                ((total_tests++))
            else
                echo -e "\t${RED}MPI implementation differs (processes=$procs).${NC}"
                ((failed_tests++))
                ((total_tests++))
            fi
            
            # Run and compare hybrid versions
            for threads in "${omp_threads[@]}"; do
                mpirun --oversubscribe -np $procs ./mpi-omp-stencil-2d $iter "input_${size}.dat" "hybrid_${size}_${iter}_${procs}_${threads}.out" $threads 0 > /dev/null
                
                echo "DIFFING MPI/OMP IMPLEMENTATION (processes=$procs, threads=$threads)"
                if diff "seq_${size}_${iter}.out" "hybrid_${size}_${iter}_${procs}_${threads}.out" > /dev/null 2>&1; then
                    echo -e "\t${GREEN}MPI/OMP matches sequential.${NC}"
                    ((total_tests++))
                else
                    echo -e "\t${RED}MPI/OMP differs (processes=$procs, threads=$threads).${NC}"
                    ((failed_tests++))
                    ((total_tests++))
                fi
            done
        done
    done
done

echo "******************************** TESTING COMPLETED ********************************"
if [ $failed_tests -eq 0 ]; then
    echo -e "\n${GREEN}All $total_tests TESTS PASSED SUCCESFULLY!${NC}"
else 
    echo -e "\n${RED}$failed_tests OUT OF $total_tests TESTS FAILED.${NC}"
fi

make clean

# Cleanup
rm -f input_*.dat *_*.out all_*.dat