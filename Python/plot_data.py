import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Set global font size
plt.rcParams.update({'font.size': 14})

def calculate_speedup(base_time, times):
    return base_time / times

def calculate_efficiency(speedup, threads):
    return speedup / threads

def calculate_serial_fraction(speedup, threads):
    return ((1 / speedup) - (1 / threads)) / (1 - (1 / threads))

def plot_data(filename):
    # Create the base 'plots' directory if it doesn't exist
    base_dir = "../plots"
    os.makedirs(base_dir, exist_ok=True)
    
    # Extract base filename (without path and extension) to include in output filenames
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Create subdirectory based on input filename
    output_dir = os.path.join(base_dir, base_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from the CSV file
    data = pd.read_csv(filename)
    
    # Determine if this is MPI data
    is_mpi = 'mpi' in filename.lower()
    x_axis_label = 'Processes' if is_mpi else 'Threads'
    parallelism_col = 'Processes' if is_mpi else 'Threads'
    
    # Extract unique matrix sizes
    unique_matrix_sizes = data['Matrix Size'].unique()
    
    # Plot T_overall, T_computation, and T_other as functions of threads/processes for each matrix size
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        
        plt.figure(figsize=(10, 6))
        plt.plot(subset[parallelism_col], subset['Time Overall (s)'], label='T_overall', marker='o')
        plt.plot(subset[parallelism_col], subset['Time Computation (s)'], label='T_computation', marker='o')
        plt.plot(subset[parallelism_col], subset['Time Other (s)'], label='T_other', marker='o')
        
        plt.xlabel(f'{x_axis_label} (p)')
        plt.ylabel('Time (seconds)')
        plt.title(f'Time Components vs {x_axis_label} for Matrix Size {matrix_size}')
        plt.legend()
        plt.grid(True)
        timing_filename = os.path.join(output_dir, f"{base_filename}_time_components_matrix_{matrix_size}.pdf")
        plt.savefig(timing_filename, bbox_inches='tight')
        print(f"Saved {timing_filename}")
        plt.close()

    # Rest of the plotting code remains exactly the same, just with updated output_dir
    # [Previous plotting code continues unchanged...]
    
    # Speedup Overall Plot
    plt.figure(figsize=(10, 6))
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        base_time_overall = subset.loc[subset[parallelism_col] == 1, 'Time Overall (s)'].values[0]
        speedup_overall = calculate_speedup(base_time_overall, subset['Time Overall (s)'])
        
        plt.plot(subset[parallelism_col], speedup_overall, label=f'Speedup Overall (n={matrix_size})', marker='o')
    
    plt.plot(subset[parallelism_col], subset[parallelism_col], label='Ideal Speedup', linestyle='--', color='gray')
    plt.xlabel(f'{x_axis_label} (p)')
    plt.ylabel('Speedup')
    plt.title(f'Speedup (Overall) vs {x_axis_label}')
    plt.legend()
    plt.grid(True)
    speedup_overall_filename = os.path.join(output_dir, f"{base_filename}_speedup_overall.pdf")
    plt.savefig(speedup_overall_filename, bbox_inches='tight')
    print(f"Saved {speedup_overall_filename}")
    plt.close()

    # Speedup Computation Plot
    plt.figure(figsize=(10, 6))
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        base_time_computation = subset.loc[subset[parallelism_col] == 1, 'Time Computation (s)'].values[0]
        speedup_computation = calculate_speedup(base_time_computation, subset['Time Computation (s)'])
        
        plt.plot(subset[parallelism_col], speedup_computation, label=f'Speedup Computation (n={matrix_size})', marker='o')
    
    plt.plot(subset[parallelism_col], subset[parallelism_col], label='Ideal Speedup', linestyle='--', color='gray')
    plt.xlabel(f'{x_axis_label} (p)')
    plt.ylabel('Speedup')
    plt.title(f'Speedup (Computation) vs {x_axis_label}')
    plt.legend()
    plt.grid(True)
    speedup_computation_filename = os.path.join(output_dir, f"{base_filename}_speedup_computation.pdf")
    plt.savefig(speedup_computation_filename, bbox_inches='tight')
    print(f"Saved {speedup_computation_filename}")
    plt.close()

    # Efficiency Overall Plot
    plt.figure(figsize=(10, 6))
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        base_time_overall = subset.loc[subset[parallelism_col] == 1, 'Time Overall (s)'].values[0]
        speedup_overall = calculate_speedup(base_time_overall, subset['Time Overall (s)'])
        efficiency_overall = calculate_efficiency(speedup_overall, subset[parallelism_col])
        
        plt.plot(subset[parallelism_col], efficiency_overall, label=f'Efficiency Overall (n={matrix_size})', marker='o')
    
    plt.axhline(1.0, label='Ideal Efficiency', linestyle='--', color='gray')
    plt.xlabel(f'{x_axis_label} (p)')
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency (Overall) vs {x_axis_label}')
    plt.legend()
    plt.grid(True)
    efficiency_overall_filename = os.path.join(output_dir, f"{base_filename}_efficiency_overall.pdf")
    plt.savefig(efficiency_overall_filename, bbox_inches='tight')
    print(f"Saved {efficiency_overall_filename}")
    plt.close()

    # Efficiency Computation Plot
    plt.figure(figsize=(10, 6))
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        base_time_computation = subset.loc[subset[parallelism_col] == 1, 'Time Computation (s)'].values[0]
        speedup_computation = calculate_speedup(base_time_computation, subset['Time Computation (s)'])
        efficiency_computation = calculate_efficiency(speedup_computation, subset[parallelism_col])
        
        plt.plot(subset[parallelism_col], efficiency_computation, label=f'Efficiency Computation (n={matrix_size})', marker='o')
    
    plt.axhline(1.0, label='Ideal Efficiency', linestyle='--', color='gray')
    plt.xlabel(f'{x_axis_label} (p)')
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency (Computation) vs {x_axis_label}')
    plt.legend()
    plt.grid(True)
    efficiency_computation_filename = os.path.join(output_dir, f"{base_filename}_efficiency_computation.pdf")
    plt.savefig(efficiency_computation_filename, bbox_inches='tight')
    print(f"Saved {efficiency_computation_filename}")
    plt.close()

    # Serial Fraction Plot e_overall
    plt.figure(figsize=(10, 6))
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        base_time_overall = subset.loc[subset[parallelism_col] == 1, 'Time Overall (s)'].values[0]
        speedup_overall = calculate_speedup(base_time_overall, subset['Time Overall (s)'])
        
        # Calculate e_overall for each thread/process count
        e_overall = calculate_serial_fraction(speedup_overall, subset[parallelism_col])
        
        # Set negative values to 0
        e_overall = e_overall.clip(lower=0)
        
        plt.plot(subset[parallelism_col], e_overall, label=f'e_overall (n={matrix_size})', marker='o')
    
    plt.xlabel(f'{x_axis_label} (p)')
    plt.ylabel('Serial Fraction (e)')
    plt.title(f'Experimentally Determined Serial Fraction (e_overall) vs {x_axis_label}')
    plt.legend()
    plt.grid(True)
    e_overall_filename = os.path.join(output_dir, f"{base_filename}_e_overall.pdf")
    plt.savefig(e_overall_filename, bbox_inches='tight')
    print(f"Saved {e_overall_filename}")
    plt.close()

    # Serial Fraction Plot e_computation
    plt.figure(figsize=(10, 6))
    for matrix_size in unique_matrix_sizes:
        subset = data[data['Matrix Size'] == matrix_size]
        base_time_computation = subset.loc[subset[parallelism_col] == 1, 'Time Computation (s)'].values[0]
        speedup_computation = calculate_speedup(base_time_computation, subset['Time Computation (s)'])
        
        # Calculate e_computation for each thread/process count
        e_computation = calculate_serial_fraction(speedup_computation, subset[parallelism_col])
        
        # Set negative values to 0
        e_computation = e_computation.clip(lower=0)
        
        plt.plot(subset[parallelism_col], e_computation, label=f'e_computation (n={matrix_size})', marker='o')
    
    plt.xlabel(f'{x_axis_label} (p)')
    plt.ylabel('Serial Fraction (e)')
    plt.title(f'Experimentally Determined Serial Fraction (e_computation) vs {x_axis_label}')
    plt.legend()
    plt.grid(True)
    e_computation_filename = os.path.join(output_dir, f"{base_filename}_e_computation.pdf")
    plt.savefig(e_computation_filename, bbox_inches='tight')
    print(f"Saved {e_computation_filename}")
    plt.close()

# Argument parsing for a single file with usage statement
parser = argparse.ArgumentParser(
    description="Plot data from a single CSV file.",
    usage=f"python3 {os.path.basename(__file__)} <data.csv>"
)
parser.add_argument("filename", type=str, nargs="?", help="Path to the CSV file containing data.")

# Check if no argument was provided; if not, show usage and exit
args = parser.parse_args()
if not args.filename:
    parser.print_usage()
    sys.exit()

# Call the plot_data function
plot_data(args.filename)