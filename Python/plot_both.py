import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_serial_fraction(speedup, total_processors):
    """Calculate experimental serial fraction using Amdahl's Law"""
    return (1/speedup - 1/total_processors)/(1 - 1/total_processors)

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
            
            # Customize the plot
            plt.xscale('log', base=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('OMP Threads')
            plt.ylabel('Serial Fraction')
            plt.title(f'{metric_type} Serial Fraction vs Threads (Matrix Size: {matrix_size})')
            plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
            
            # Set x-ticks to match the actual thread counts
            thread_counts = sorted(df['OMP Threads'].unique())
            plt.xticks(thread_counts, thread_counts)
            
            # Add gridlines
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 
                       f'serial_fraction_{metric_type.lower()}_matrix_{matrix_size}.png'), 
                       dpi=300, bbox_inches='tight')
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
            plt.savefig(os.path.join(output_dir, 
                       f'time_{timing_type.lower()}_matrix_{matrix_size}.png'), 
                       dpi=300, bbox_inches='tight')
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
                plt.savefig(os.path.join(output_dir, 
                           f'{plot_type.lower()}_matrix_{matrix_size}_{metric_type.lower()}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()

def main():
    output_dir = '../Plots/mpi-omp-hybrid-plots'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv('../data/mpi-omp.csv')

    plot_speedup_efficiency(df, output_dir)
    plot_timing_components(df, output_dir)
    plot_serial_fraction(df, output_dir)
    print(f"All plots have been generated in {output_dir}!")

if __name__ == "__main__":
    main()