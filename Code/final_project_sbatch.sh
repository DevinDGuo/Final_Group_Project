#!/bin/bash
#SBATCH --job-name="Group_Final"
#SBATCH --output="Group_Final.%j.%N.out"
#SBATCH --mail-user=ddguo@coastal.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=compute
#SBATCH --nodes=16                # 16 nodes
#SBATCH --ntasks-per-node=1          # 1 processes per node
#SBATCH --cpus-per-task=128          # 128 threads per process
#SBATCH --mem=200G
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 06:00:00

# Load required modules
module load cpu/0.17.3b gcc/10.2.0/npcyll4 openmpi/4.1.1
module load python3

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the Python script
python3 ../Python/gather_data_all.py pthread.csv omp.csv mpi.csv mpi-omp.csv