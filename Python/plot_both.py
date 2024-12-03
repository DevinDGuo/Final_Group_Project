import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Set global font size
plt.rcParams.update({'font.size': 14})

def calculate_serial_fraction(speedup, total_processors):
    """Calculate experimental serial fraction using Amdahl's Law"""
    return (1/speedup - 1/total_processors)/(1 - 1/total_processors)

def create_output_directory(input_file):
    """Create output directory structure based on input filename"""
    base_dir = "../plots"
    os.makedirs(base_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(base_dir, base_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir, base_filename

def plot_serial_fraction(df, output_dir):
    """Create line plots showing the serial fraction for each configuration."""
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
    ]
    markers = ['o', 's', '^', 'D', 'v']
    
    matrix_sizes = sorted(df['Matrix Size'].unique())
    process_counts = sorted(df['Processes'].unique())
    
    for matrix_size in matrix_sizes:
        for metric_type in ['Overall', 'Computation']:
            plt.figure(figsize=(12, 8))
            
            for idx, process_count in enumerate(process_counts):
                data = df[(df['Matrix Size'] == matrix_size) & 
                         (df['Processes'] == process_count)]
                
                # Get baseline time (serial case)
                if metric_type == 'Overall':
                    baseline = df[(df['Matrix Size'] == matrix_size) & 
                                (df['Processes'] == 1) & 
                                (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
                    current_times = data['Time Overall (s)']
                else:  # Computation
                    baseline = df[(df['Matrix Size'] == matrix_size) & 
                                (df['Processes'] == 1) & 
                                (df['OMP Threads'] == 1)]['Time Computation (s)'].iloc[0]
                    current_times = data['Time Computation (s)']
                
                # Calculate speedup and serial fraction
                speedup = baseline / current_times
                total_processors = process_count * data['OMP Threads']
                serial_fraction = calculate_serial_fraction(speedup, total_processors)
                
                # Plot
                plt.plot(data['OMP Threads'], serial_fraction, 
                        marker=markers[idx % len(markers)],
                        linestyle='-', 
                        linewidth=2,
                        markersize=8,
                        label=f'{process_count} Processes',
                        color=colors[idx % len(colors)])
            
            plt.xscale('log', base=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('OMP Threads')
            plt.ylabel('Serial Fraction')
            plt.title(f'{metric_type} Serial Fraction vs Threads (Matrix Size: {matrix_size})')
            plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
            
            thread_counts = sorted(df['OMP Threads'].unique())
            plt.xticks(thread_counts, thread_counts)
            
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'serial_fraction_{metric_type.lower()}_matrix_{matrix_size}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {save_path}")
            plt.close()

def plot_timing_components(df, output_dir):
    """Create line plots showing the timing components for each configuration."""
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
    ]
    markers = ['o', 's', '^', 'D', 'v']
    
    matrix_sizes = sorted(df['Matrix Size'].unique())
    process_counts = sorted(df['Processes'].unique())
    
    for matrix_size in matrix_sizes:
        for timing_type in ['Overall', 'Computation', 'Other']:
            plt.figure(figsize=(12, 8))
            
            for idx, process_count in enumerate(process_counts):
                data = df[(df['Matrix Size'] == matrix_size) & 
                         (df['Processes'] == process_count)]
                
                if timing_type == 'Overall':
                    times = data['Time Overall (s)']
                elif timing_type == 'Computation':
                    times = data['Time Computation (s)']
                else:  # Other
                    times = data['Time Other (s)']
                
                plt.plot(data['OMP Threads'], times, 
                        marker=markers[idx % len(markers)],
                        linestyle='-', 
                        linewidth=2,
                        markersize=8,
                        label=f'{process_count} Processes',
                        color=colors[idx % len(colors)])
            
            plt.xscale('log', base=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('OMP Threads')
            plt.ylabel(f'{timing_type} Time (seconds)')
            plt.title(f'{timing_type} Time vs Threads (Matrix Size: {matrix_size})')
            plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
            
            thread_counts = sorted(df['OMP Threads'].unique())
            plt.xticks(thread_counts, thread_counts)
            
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'time_{timing_type.lower()}_matrix_{matrix_size}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {save_path}")
            plt.close()

def plot_speedup_efficiency(df, output_dir):
    """Create speedup and efficiency plots for different matrix sizes and metrics."""
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
    ]
    markers = ['o', 's', '^', 'D', 'v']

    matrix_sizes = sorted(df['Matrix Size'].unique())
    process_counts = sorted(df['Processes'].unique())

    for matrix_size in matrix_sizes:
        for metric_type in ['Overall', 'Computation']:
            for plot_type in ['Speedup', 'Efficiency']:
                plt.figure(figsize=(12, 8))
                
                max_actual_speedup = 0
                
                for idx, process_count in enumerate(process_counts):
                    data = df[(df['Matrix Size'] == matrix_size) & 
                             (df['Processes'] == process_count)]
                    
                    if metric_type == 'Overall':
                        baseline = df[(df['Matrix Size'] == matrix_size) & 
                                    (df['Processes'] == 1) & 
                                    (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
                        current_times = data['Time Overall (s)']
                    else:  # Computation
                        baseline = df[(df['Matrix Size'] == matrix_size) & 
                                    (df['Processes'] == 1) & 
                                    (df['OMP Threads'] == 1)]['Time Computation (s)'].iloc[0]
                        current_times = data['Time Computation (s)']
                    
                    speedup = baseline / current_times
                    max_actual_speedup = max(max_actual_speedup, speedup.max())
                    
                    if plot_type == 'Speedup':
                        y_values = speedup
                    else:  # Efficiency
                        total_processors = process_count * data['OMP Threads']
                        y_values = speedup / total_processors
                    
                    plt.plot(data['OMP Threads'], y_values, 
                            marker=markers[idx % len(markers)],
                            linestyle='-', 
                            linewidth=2,
                            markersize=8,
                            label=f'{process_count} Processes',
                            color=colors[idx % len(colors)])
                
                # Add ideal lines
                thread_values = sorted(df['OMP Threads'].unique())
                if plot_type == 'Speedup':
                    # Single reference line
                    ideal_speedup = thread_values  # Linear scaling with number of threads
                    plt.plot(thread_values, ideal_speedup,
                            linestyle='--',
                            color='black',
                            label='Ideal',
                            alpha=0.5)
                elif plot_type == 'Efficiency':
                    # Add horizontal line at y=1 for perfect efficiency
                    plt.axhline(y=1, color='black', linestyle='--', 
                                label='Ideal Efficiency', alpha=0.5)
                
                if plot_type == 'Speedup':
                    plt.ylim(0, max_actual_speedup * 1.2)
                else:  # Efficiency
                    plt.ylim(0, 1.2)
                
                plt.xscale('log', base=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('OMP Threads')
                plt.ylabel(plot_type)
                plt.title(f'{metric_type} {plot_type} vs Threads (Matrix Size: {matrix_size})')
                plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
                
                thread_counts = sorted(df['OMP Threads'].unique())
                plt.xticks(thread_counts, thread_counts)
                
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.tight_layout()
                # Modify the save_path line in plot_speedup_efficiency:
                save_path = os.path.join(output_dir, 
                    f'{plot_type.lower()}_matrix{matrix_size}_{metric_type.lower()}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved {save_path}")
                plt.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Plot data from a single CSV file.",
        usage=f"python3 {os.path.basename(__file__)} <data.csv>"
    )
    parser.add_argument("filename", type=str, nargs="?", help="Path to the CSV file containing data.")
    return parser

def main():
    # Parse command line arguments
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Check if no argument was provided
    if not args.filename:
        parser.print_usage()
        sys.exit(1)

    # Create output directory structure
    output_dir, base_filename = create_output_directory(args.filename)

    try:
        # Read the CSV file
        df = pd.read_csv(args.filename)
    except FileNotFoundError:
        print(f"Error: Input file '{args.filename}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{args.filename}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        sys.exit(1)

    # Generate all plots
    plot_speedup_efficiency(df, output_dir)
    plot_timing_components(df, output_dir)
    plot_serial_fraction(df, output_dir)
    print(f"All plots have been generated in {output_dir}!")

if __name__ == "__main__":
    main()