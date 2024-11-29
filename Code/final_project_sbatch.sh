#!/bin/bash
#SBATCH --job-name="Group_Final"
#SBATCH --output="Group_Final.%j.%N.out"
#SBATCH --mail-user=ddguo@coastal.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=compute
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=20G
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 05:00:00

module load cpu/0.17.3b gcc/10.2.0/npcyll4 openmpi/4.1.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -np 16 --map-by ppr:2:node:PE=64 ./mpi-omp-stencil-2d 12 A.dat B.dat 64 0

module load python3
python3 ../Python/gather_data_all.py pthread.csv omp.csv mpi.csv mpi-omp.csv