import os
import subprocess
import sys

# Check for correct arguments
if len(sys.argv) != 3:
    print("Usage: python3 gather_data.py <pthread_csv_file> <omp_csv_file>")
    sys.exit(1)

# Data directory path
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# Output CSV file paths
PTHREAD_CSV = os.path.join(DATA_DIR, sys.argv[1])
OMP_CSV = os.path.join(DATA_DIR, sys.argv[2])

# Change to the 'code' directory
os.chdir("../code")

# Clean and compile the project
subprocess.run(["make", "clean", "all"])

# Matrix sizes (n) and thread counts (t)
MATRIX_SIZES = [5000, 10000, 20000, 40000]
THREAD_COUNTS = [1, 2, 4, 8, 16]

# Number of time steps
TS = 12

# Input and output file names
INPUT_FILE = "initial.dat"
OUTPUT_FILE = "final.dat"

# Initialize CSV files with headers
with open(PTHREAD_CSV, "w") as f:
    f.write("Matrix Size,Threads,Time Overall (s),Time Computation (s),Time Other (s)\n")
with open(OMP_CSV, "w") as f:
    f.write("Matrix Size,Threads,Time Overall (s),Time Computation (s),Time Other (s)\n")

# Loop over each matrix size
for n in MATRIX_SIZES:
    # Create the input data file for the current matrix size
    subprocess.run(["./make-2d", str(n), str(n), INPUT_FILE])

    # Loop over each thread count
    for t in THREAD_COUNTS:
        print(f"Running {TS} iterations for matrix size {n}x{n} with {t} threads...")

        # Run pthread stencil and capture output
        result = subprocess.run(["./pth-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, "0", str(t)], capture_output=True, text=True)
        PTHREAD_OUTPUT = result.stdout

        # Parse timing values from the output
        PTHREAD_TIME_OVERALL = next((line.split()[2] for line in PTHREAD_OUTPUT.splitlines() if "Time Overall" in line), "N/A")
        PTHREAD_TIME_COMP = next((line.split()[2] for line in PTHREAD_OUTPUT.splitlines() if "Time Computation" in line), "N/A")
        PTHREAD_TIME_OTHER = next((line.split()[2] for line in PTHREAD_OUTPUT.splitlines() if "Time Other" in line), "N/A")

        # Append parsed results to CSV
        with open(PTHREAD_CSV, "a") as f:
            f.write(f"{n},{t},{PTHREAD_TIME_OVERALL},{PTHREAD_TIME_COMP},{PTHREAD_TIME_OTHER}\n")

        # Run OpenMP stencil and capture output
        result = subprocess.run(["./omp-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, "0", str(t)], capture_output=True, text=True)
        OMP_OUTPUT = result.stdout

        # Parse timing values from the output
        OMP_TIME_OVERALL = next((line.split()[2] for line in OMP_OUTPUT.splitlines() if "Time Overall" in line), "N/A")
        OMP_TIME_COMP = next((line.split()[2] for line in OMP_OUTPUT.splitlines() if "Time Computation" in line), "N/A")
        OMP_TIME_OTHER = next((line.split()[2] for line in OMP_OUTPUT.splitlines() if "Time Other" in line), "N/A")

        # Append parsed results to CSV
        with open(OMP_CSV, "a") as f:
            f.write(f"{n},{t},{OMP_TIME_OVERALL},{OMP_TIME_COMP},{OMP_TIME_OTHER}\n")

# Clean up after execution
subprocess.run(["make", "clean"])

print(f"Experiments completed. Results saved to {PTHREAD_CSV} and {OMP_CSV}.")
