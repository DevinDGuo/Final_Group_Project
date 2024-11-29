import os
import subprocess
import sys

# Check for correct arguments
if len(sys.argv) != 3:
    print("Usage: python3 gather_data.py <mpi_csv_file> <mpi_omp_csv_file>")
    sys.exit(1)

# Data directory path
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# Output CSV file paths
MPI_CSV = os.path.join(DATA_DIR, sys.argv[1])
MPI_OMP_CSV = os.path.join(DATA_DIR, sys.argv[2])

# Change to the 'code' directory
os.chdir("../Code")

# Clean and compile the project
subprocess.run(["make", "clean", "all"])

# Matrix sizes (n) and process counts
MATRIX_SIZES = [5000, 10000, 20000, 40000]
PROCESS_COUNTS = [1, 2, 4, 8, 16]

# Number of time steps
TS = 12

# Input and output file names
INPUT_FILE = "initial.dat"
OUTPUT_FILE = "final.dat"

# Initialize CSV files with headers
for csv_file, header in zip(
    [MPI_CSV, MPI_OMP_CSV],
    [
        "Matrix Size,Processes,Time Overall (s),Time Computation (s),Time Other (s)\n",
        "Matrix Size,Processes,OMP Threads,Time Overall (s),Time Computation (s),Time Other (s)\n",
    ],
):
    with open(csv_file, "w") as f:
        f.write(header)

# Loop over each matrix size
for n in MATRIX_SIZES:
    # Create the input data file for the current matrix size
    subprocess.run(["./make-2d", str(n), str(n), INPUT_FILE])

    # Run MPI stencil
    for p in PROCESS_COUNTS:
        print(f"Running MPI driver: {TS} iterations for matrix size {n}x{n} with {p} processes...")
        result = subprocess.run(
            ["mpirun", "-np", str(p), "./mpi-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, "0"],
            capture_output=True,
            text=True,
        )
        MPI_OUTPUT = result.stdout
        MPI_TIME_OVERALL = next((line.split()[2] for line in MPI_OUTPUT.splitlines() if "Time Overall" in line), "N/A")
        MPI_TIME_COMP = next((line.split()[2] for line in MPI_OUTPUT.splitlines() if "Time Computation" in line), "N/A")
        MPI_TIME_OTHER = next((line.split()[2] for line in MPI_OUTPUT.splitlines() if "Time Other" in line), "N/A")
        with open(MPI_CSV, "a") as f:
            f.write(f"{n},{p},{MPI_TIME_OVERALL},{MPI_TIME_COMP},{MPI_TIME_OTHER}\n")

    # Run MPI-OMP stencil
    for p in PROCESS_COUNTS:
        for t in [1, 2, 4, 8, 16, 64]:
            print(f"Running MPI-OMP driver: {TS} iterations for matrix size {n}x{n} with {p} processes and {t} threads...")
            result = subprocess.run(
                ["mpirun", "-np", str(p), "./mpi-omp-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, str(t), "0"],
                capture_output=True,
                text=True,
            )
            MPI_OMP_OUTPUT = result.stdout
            MPI_OMP_TIME_OVERALL = next(
                (line.split()[2] for line in MPI_OMP_OUTPUT.splitlines() if "Time Overall" in line), "N/A"
            )
            MPI_OMP_TIME_COMP = next(
                (line.split()[2] for line in MPI_OMP_OUTPUT.splitlines() if "Time Computation" in line), "N/A"
            )
            MPI_OMP_TIME_OTHER = next(
                (line.split()[2] for line in MPI_OMP_OUTPUT.splitlines() if "Time Other" in line), "N/A"
            )
            with open(MPI_OMP_CSV, "a") as f:
                f.write(f"{n},{p},{t},{MPI_OMP_TIME_OVERALL},{MPI_OMP_TIME_COMP},{MPI_OMP_TIME_OTHER}\n")

# Clean up after execution
subprocess.run(["make", "clean"])

print(f"Experiments completed. Results saved to {MPI_CSV} and {MPI_OMP_CSV}.")