#!/bin/bash
#SBATCH --job-name="group_final"
#SBATCH --output="group_final.%j.%N.out"
#SBATCH --mail-user=ddguo@coastal.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=60G
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 00:30:00

# Run the job by calling your script
python3 ../python/gather_data.py pthread.csv omp.csv mpi.csv mpi-omp.csv