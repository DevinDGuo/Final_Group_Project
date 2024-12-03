import os
import subprocess
import sys

print("Collecting data...")
subprocess.run(["python3", "gather_data.py", "pthreads_sample.csv", "omp_sample.csv", "mpi_sample.csv", "mpi_omp_sample.csv"])
print("Data Gathering Completed")

print("Plotting data...")
print("Plotting data for pthreads...")
subprocess.run(["python3", "plot_data.py", "../data/pthreads.csv"])
print("Plotting data for omp...")
subprocess.run(["python3", "plot_data.py", "../data/omp.csv"])
print("Plotting completed.")
print("Plotting data for mpi...")
subprocess.run(["python3", "plot_data.py", "../data/mpi.csv"])
print("Plotting completed.")
print("Plotting data for mpi-omp-hybrid...")
subprocess.run(["python3", "plot_both.py"])
print("Plotting completed."

print("Run all completed.")