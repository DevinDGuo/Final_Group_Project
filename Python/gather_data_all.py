import os
import subprocess
import sys

# Check for correct arguments
if len(sys.argv) != 5:
    print("Usage: python3 gather_data.py <pthread_csv_file> <omp_csv_file> <mpi_csv_file> <mpi_omp_csv_file>")
    sys.exit(1)

# Data directory path
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# Output CSV file paths
PTHREAD_CSV = os.path.join(DATA_DIR, sys.argv[1])
OMP_CSV = os.path.join(DATA_DIR, sys.argv[2])
MPI_CSV = os.path.join(DATA_DIR, sys.argv[3])
MPI_OMP_CSV = os.path.join(DATA_DIR, sys.argv[4])

# Change to the 'code' directory
os.chdir("../Code")

# Clean and compile the project
try:
    subprocess.run(["make", "clean", "all"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error during compilation: {e}")
    sys.exit(1)

# Matrix sizes (n) and thread/process counts
MATRIX_SIZES = [5000, 10000, 20000, 40000]
THREAD_COUNTS = [1, 2, 4, 8, 16]        # Regular OpenMP and Pthread tests
PROCESS_COUNTS = [1, 2, 4, 8, 16]       # MPI processes
THREAD_COUNTS_HYBRID = [1, 2, 4, 8, 16, 32, 64, 128]  # Hybrid MPI+OpenMP can go up to 128

# Number of time steps
TS = 12

# Input and output file names
INPUT_FILE = "initial.dat"
OUTPUT_FILE = "final.dat"

# Initialize CSV files with headers
for csv_file, header in zip(
    [PTHREAD_CSV, OMP_CSV, MPI_CSV, MPI_OMP_CSV],
    [
        "Matrix Size,Threads,Time Overall (s),Time Computation (s),Time Other (s)\n",
        "Matrix Size,Threads,Time Overall (s),Time Computation (s),Time Other (s)\n",
        "Matrix Size,Processes,Time Overall (s),Time Computation (s),Time Other (s)\n",
        "Matrix Size,Processes,OMP Threads,Time Overall (s),Time Computation (s),Time Other (s)\n",
    ],
):
    with open(csv_file, "w") as f:
        f.write(header)

# Loop over each matrix size
for n in MATRIX_SIZES:
    try:
        # Create the input data file for the current matrix size
        subprocess.run(["./make-2d", str(n), str(n), INPUT_FILE], check=True)

        # Run pthread stencil
        for t in THREAD_COUNTS:
            print(f"Running PTHREAD driver: {TS} iterations for matrix size {n}x{n} with {t} threads...")
            try:
                result = subprocess.run(
                    ["./pth-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, "0", str(t)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                PTHREAD_OUTPUT = result.stdout
                PTHREAD_TIME_OVERALL = next((line.split()[2] for line in PTHREAD_OUTPUT.splitlines() if "Time Overall" in line), "N/A")
                PTHREAD_TIME_COMP = next((line.split()[2] for line in PTHREAD_OUTPUT.splitlines() if "Time Computation" in line), "N/A")
                PTHREAD_TIME_OTHER = next((line.split()[2] for line in PTHREAD_OUTPUT.splitlines() if "Time Other" in line), "N/A")
                with open(PTHREAD_CSV, "a") as f:
                    f.write(f"{n},{t},{PTHREAD_TIME_OVERALL},{PTHREAD_TIME_COMP},{PTHREAD_TIME_OTHER}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error running pthread with {t} threads: {e}")
                continue

        # Run OpenMP stencil
        for t in THREAD_COUNTS:
            print(f"Running OMP driver: {TS} iterations for matrix size {n}x{n} with {t} threads...")
            try:
                result = subprocess.run(
                    ["./omp-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, "0", str(t)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                OMP_OUTPUT = result.stdout
                OMP_TIME_OVERALL = next((line.split()[2] for line in OMP_OUTPUT.splitlines() if "Time Overall" in line), "N/A")
                OMP_TIME_COMP = next((line.split()[2] for line in OMP_OUTPUT.splitlines() if "Time Computation" in line), "N/A")
                OMP_TIME_OTHER = next((line.split()[2] for line in OMP_OUTPUT.splitlines() if "Time Other" in line), "N/A")
                with open(OMP_CSV, "a") as f:
                    f.write(f"{n},{t},{OMP_TIME_OVERALL},{OMP_TIME_COMP},{OMP_TIME_OTHER}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error running OpenMP with {t} threads: {e}")
                continue

        # Run MPI stencil
        for p in PROCESS_COUNTS:
            print(f"Running MPI driver: {TS} iterations for matrix size {n}x{n} with {p} processes...")
            try:
                result = subprocess.run(
                    ["mpirun", "-np", str(p), "--map-by", "node", "--bind-to", "core",
                     "./mpi-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, "0"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                MPI_OUTPUT = result.stdout
                MPI_TIME_OVERALL = next((line.split()[2] for line in MPI_OUTPUT.splitlines() if "Time Overall" in line), "N/A")
                MPI_TIME_COMP = next((line.split()[2] for line in MPI_OUTPUT.splitlines() if "Time Computation" in line), "N/A")
                MPI_TIME_OTHER = next((line.split()[2] for line in MPI_OUTPUT.splitlines() if "Time Other" in line), "N/A")
                with open(MPI_CSV, "a") as f:
                    f.write(f"{n},{p},{MPI_TIME_OVERALL},{MPI_TIME_COMP},{MPI_TIME_OTHER}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error running MPI with {p} processes: {e}")
                continue

        # Run MPI-OMP stencil
        for p in PROCESS_COUNTS:
            for t in THREAD_COUNTS_HYBRID:
                print(f"Running MPI-OMP driver: {TS} iterations for matrix size {n}x{n} with {p} processes and {t} threads...")
                try:
                    result = subprocess.run(
                        ["mpirun", "-np", str(p), "--map-by", f"node:PE={t}", "--bind-to", "core",
                         "./mpi-omp-stencil-2d", str(TS), INPUT_FILE, OUTPUT_FILE, str(t), "0"],
                        capture_output=True,
                        text=True,
                        check=True
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
                except subprocess.CalledProcessError as e:
                    print(f"Error running MPI-OMP with {p} processes and {t} threads: {e}")
                    continue

    except Exception as e:
        print(f"Error processing matrix size {n}: {e}")
        continue

# Clean up after execution
try:
    subprocess.run(["make", "clean"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error during cleanup: {e}")

print(f"Experiments completed. Results saved to {PTHREAD_CSV}, {OMP_CSV}, {MPI_CSV}, and {MPI_OMP_CSV}.")