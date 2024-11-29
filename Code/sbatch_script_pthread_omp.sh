#!/bin/bash
#SBATCH --job-name="performance_data"
#SBATCH --output="performance_data.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 00:40:00

# Run the job by calling your script
python3 ../python/gather_data.py pthread.csv omp.csv