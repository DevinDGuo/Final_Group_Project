# 2D Stencil Computation Analysis

This project implements and analyzes different parallel computing approaches (OpenMP, Pthreads, MPI, and hybrid MPI+OpenMP) for 2D stencil computations.

## Project Structure

```
final_project/
├── code/
│   ├── Makefile                     # Build configuration
│   ├── MyMPI.c                      # MPI utility functions implementation
│   ├── MyMPI.h                      # MPI utility functions header
│   ├── final_project_sbatch.sh      # SLURM job submission script
│   ├── make-2d.c                    # Input data generator
│   ├── mpi-omp-stencil-2d.c         # Hybrid MPI+OpenMP implementation
│   ├── mpi-stencil-2d.c             # MPI implementation
│   ├── mpi_run_test.bash            # MPI testing script
│   ├── my_barrier.c                 # Custom barrier implementation
│   ├── my_barrier.h                 # Custom barrier header
│   ├── omp-stencil-2d.c             # OpenMP implementation
│   ├── print-2d.c                   # Matrix visualization utility
│   ├── pth-stencil-2d.c             # Pthreads implementation
│   ├── sbatch_script_pthread_omp.sh # SLURM script for pthread/OMP
│   ├── stencil-2d.c                 # Serial implementation
│   ├── test_all_versions.sh         # Validation testing script
│   ├── timer.h                      # Timing utilities
│   ├── utilities.c                  # Core utilities implementation
│   └── utilities.h                  # Core utilities header
├── data/
│   ├── mpi-omp.csv                  # Hybrid implementation timing data
│   ├── mpi.csv                      # MPI implementation timing data
│   ├── omp.csv                      # OpenMP implementation timing data
│   └── pthread.csv                  # Pthreads implementation timing data
├── plots/
│   └── *.png                        # Generated visualizations
└── python/
    ├── gather_data_all.py           # Comprehensive data collection
    ├── make-movie.py                # Stencil computation visualization
    ├── plot_both.py                 # Hybrid implementation analysis
    └── plot_data.py                 # Individual implementation analysis
```

## Dependencies

### Compilation Requirements
- GCC compiler with OpenMP support
    - *Note: If compiling on MacOS, gcc version 14 must be specified.  
    If using mpicc, locate the MPI wrapper configuration file and set `compiler=gcc-14`*
- POSIX threads library
- MPI implementation (OpenMPI recommended)
- Make build system

### Python Requirements
- Python 3.6+
- pandas
- matplotlib
- numpy
- ffmpeg (for movie generation)

Install Python dependencies:
```bash
pip install pandas matplotlib numpy
```

## Building the Project

Navigate to the code directory and build all executables:
```bash
cd code
make clean all
```

## Running the Programs

### Serial Version
```bash
./stencil-2d <iterations> <input_file> <output_file> <all-iterations-file>
```

### OpenMP Version
```bash
./omp-stencil-2d <iterations> <input_file> <output_file> <debug_level> <num_threads> [all-iterations-file]
```

### Pthreads Version
```bash
./pth-stencil-2d <iterations> <input_file> <output_file> <debug_level> <num_threads> [all-iterations-file]
```

### MPI Version
```bash
mpirun -np <processes> ./mpi-stencil-2d <iterations> <input_file> <output_file> <debug_level> [all-iterations-file]
```

### Hybrid MPI+OpenMP Version
```bash
mpirun -np <processes> ./mpi-omp-stencil-2d <iterations> <input_file> <output_file> <num_threads> <debug_level> [all-iterations-file]
```

## Data Collection and Analysis

### Generating Input Data
```bash
./make-2d <rows> <cols> <output_file>
```

### Running Performance Tests
Use the SLURM scripts:
```bash
sbatch final_project_sbatch.sh
```

### Analyzing Results
Navigate to the python directory:

1. Generate plots for individual implementations:
```bash
python plot_data.py ../data/<implementation>.csv
```

2. Generate hybrid analysis plots:
```bash
python plot_both.py
```

3. Create visualization of the stencil computation:
```bash
python make-movie.py <input_data_file> <output_movie.mp4>
```

## File Descriptions

### Core Implementation Files
- `utilities.c/h`: Core utilities for matrix operations and parallel implementations
- `my_barrier.c/h`: Custom barrier implementation for thread synchronization
- `MyMPI.c/h`: MPI utility functions for matrix operations
- `timer.h`: High-precision timing utilities

### Executable Source Files
- `stencil-2d.c`: Serial implementation
- `omp-stencil-2d.c`: OpenMP implementation
- `pth-stencil-2d.c`: Pthreads implementation
- `mpi-stencil-2d.c`: MPI implementation
- `mpi-omp-stencil-2d.c`: Hybrid MPI+OpenMP implementation
- `make-2d.c`: Input data generator
- `print-2d.c`: Matrix visualization utility

### Analysis Scripts
- `gather_data_all.py`: Comprehensive data collection script
- `plot_data.py`: Individual implementation analysis
- `plot_both.py`: Hybrid implementation analysis
- `make-movie.py`: Stencil computation visualization

### Build and Run Scripts
- `Makefile`: Build configuration
- `final_project_sbatch.sh`: SLURM job submission script
- `test_all_versions.sh`: Validation testing script
- `mpi_run_test.bash`: MPI testing script

## Debug Levels
- 0: Minimal output (timing only)
- 1: Basic progress information
- 2: Detailed debug information including matrix states