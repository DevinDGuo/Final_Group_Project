#!/bin/bash
#SBATCH --job-name="Group_Final"
#SBATCH --output="Group_Final.%j.%N.out"
#SBATCH --mail-user=ddguo@coastal.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=250G
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 02:15:00

module load cpu/0.17.3b  gcc/10.2.0/npcyll4 openmpi/4.1.1

module load python3

# Run the job by calling your script
python3 ../Python/gather_data_all.py pthread.csv omp.csv mpi.csv mpi-omp.csv