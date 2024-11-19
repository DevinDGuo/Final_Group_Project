import os
import subprocess
import sys

print("Collecting data...")
subprocess.run(["python3", "gather_data.py", "pthreads.csv", "omp.csv"])
print("Data Gathering Completed")

print("Plotting data...")
print("Plotting data for pthreads...")
subprocess.run(["python3", "plot_data.py", "../data/pthreads.csv"])
print("Plotting data for omp...")
subprocess.run(["python3", "plot_data.py", "../data/omp.csv"])
print("Plotting completed.")

print("Run all completed.")