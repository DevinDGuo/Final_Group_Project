import pandas as pd
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt

def create_best_config_plots(df, output_dir, plot_type):
    """Create plots showing just the best performing configuration for each matrix size"""
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311']  # One color per matrix size
    markers = ['o', 's', '^', 'D']
    matrix_sizes = sorted(df['Matrix Size'].unique())
    
    plt.figure(figsize=(12, 8))
    
    for idx, matrix_size in enumerate(matrix_sizes):
        baseline = df[(df['Matrix Size'] == matrix_size) & 
                     (df['Processes'] == 1) & 
                     (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
        
        matrix_data = df[df['Matrix Size'] == matrix_size].copy()
        matrix_data['Speedup'] = baseline / matrix_data['Time Overall (s)']
        
        # Find best configuration based on speedup
        best_idx = matrix_data['Speedup'].idxmax()
        best_config = matrix_data.loc[best_idx]
        
        # Get data for the best process count
        best_process_data = df[(df['Matrix Size'] == matrix_size) & 
                             (df['Processes'] == best_config['Processes'])]
        speedups = baseline / best_process_data['Time Overall (s)']
        
        if plot_type == 'speedup':
            y_values = speedups
            ylabel = 'Speedup'
            title = 'Best Process Configuration Speedup per Matrix Size'
        elif plot_type == 'efficiency':
            total_processors = best_config['Processes'] * best_process_data['OMP Threads']
            y_values = speedups / total_processors
            ylabel = 'Efficiency'
            title = 'Best Process Configuration Efficiency per Matrix Size'
        else:  # serial fraction
            total_processors = best_config['Processes'] * best_process_data['OMP Threads']
            y_values = (1/speedups - 1/total_processors)/(1 - 1/total_processors)
            ylabel = 'Serial Fraction'
            title = 'Best Process Configuration Serial Fraction per Matrix Size'
        
        plt.plot(best_process_data['OMP Threads'], y_values,
                marker=markers[idx % len(markers)],
                linestyle='-',
                linewidth=2,
                markersize=8,
                label=f'Matrix {matrix_size} (P={int(best_config["Processes"])})',
                color=colors[idx % len(colors)])
    
    plt.xscale('log', base=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('OMP Threads')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
    
    thread_counts = sorted(df['OMP Threads'].unique())
    plt.xticks(thread_counts, thread_counts)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"best_configs_{plot_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def find_best_overall_config(df):
    """Find the process-thread combination that performs best across all matrix sizes"""
    process_thread_scores = {}
    
    # For each process-thread combination, calculate average speedup across matrix sizes
    for process in df['Processes'].unique():
        for thread in df['OMP Threads'].unique():
            speedups = []
            for matrix_size in df['Matrix Size'].unique():
                baseline = df[(df['Matrix Size'] == matrix_size) & 
                            (df['Processes'] == 1) & 
                            (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
                
                try:
                    current_time = df[(df['Matrix Size'] == matrix_size) & 
                                    (df['Processes'] == process) & 
                                    (df['OMP Threads'] == thread)]['Time Overall (s)'].iloc[0]
                    speedup = baseline / current_time
                    speedups.append(speedup)
                except IndexError:
                    continue
                
            if speedups:
                avg_speedup = np.mean(speedups)
                process_thread_scores[(process, thread)] = avg_speedup
    
    # Find the best performing configuration
    best_config = max(process_thread_scores.items(), key=lambda x: x[1])
    return best_config[0][0], best_config[0][1], best_config[1]
    
def analyze_speedups(df, output_dir):
    matrix_sizes = sorted(df['Matrix Size'].unique())
    
    print("Best Speedups for each Matrix Size:")
    print("-" * 50)
    
    for matrix_size in matrix_sizes:
        baseline = df[(df['Matrix Size'] == matrix_size) & 
                     (df['Processes'] == 1) & 
                     (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
        
        matrix_data = df[df['Matrix Size'] == matrix_size].copy()
        matrix_data['Speedup'] = baseline / matrix_data['Time Overall (s)']
        
        best_idx = matrix_data['Speedup'].idxmax()
        best_config = matrix_data.loc[best_idx]
        
        print(f"\nMatrix Size: {matrix_size}")
        print(f"Best Speedup: {best_config['Speedup']:.2f}x")
        print(f"Configuration: {int(best_config['Processes'])} processes, {int(best_config['OMP Threads'])} threads")
        print(f"Time: {best_config['Time Overall (s)']:.2f} seconds")
    
    # Create all three plots
    create_best_config_plots(df, output_dir, 'speedup')
    create_best_config_plots(df, output_dir, 'efficiency')
    create_best_config_plots(df, output_dir, 'serial_fraction')
    
    # Find best overall process-thread combination
    best_processes, best_threads, avg_speedup = find_best_overall_config(df)
    
    print("\n" + "=" * 50)
    print(f"Best Overall Configuration:")
    print(f"Processes: {int(best_processes)}")
    print(f"Threads: {int(best_threads)}")
    print(f"Average Speedup across all matrix sizes: {avg_speedup:.2f}x")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze speedup data from CSV file.",
        usage=f"python3 {os.path.basename(__file__)} <data.csv>"
    )
    parser.add_argument("filename", type=str, help="Path to the CSV file containing data.")
    return parser

def main():
    parser = parse_arguments()
    args = parser.parse_args()

    try:
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
    
    output_dir = "../plots/analysis_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    analyze_speedups(df, output_dir)

if __name__ == "__main__":
    main()
