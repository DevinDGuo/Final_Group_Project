import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, AutoLocator
import matplotlib.ticker
import os
import argparse
import sys
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze performance data from parallel implementations.')
    parser.add_argument('--omp', required=True, help='Path to OpenMP CSV file')
    parser.add_argument('--pthread', required=True, help='Path to Pthread CSV file')
    parser.add_argument('--mpi', required=True, help='Path to MPI CSV file')
    parser.add_argument('-o', '--output', default='plots', 
                        help='Directory to save output plots (default: plots)')
    return parser.parse_args()

def format_time_ticks(x, p):
    """Format time values for tick labels"""
    if x < 1:
        return f'{x:.2f}'
    elif x < 10:
        return f'{x:.1f}'
    else:
        return f'{int(x)}'

def prepare_data(omp_file, pthread_file, mpi_file):
    """Prepare and combine data from all implementation files"""
    try:
        # Read each CSV file
        omp_df = pd.read_csv(omp_file)
        pthread_df = pd.read_csv(pthread_file)
        mpi_df = pd.read_csv(mpi_file)
        
        # Rename columns to match expected format
        column_mapping = {
            'Matrix Size': 'ROWS',
            'Threads': 'P',
            'Processes': 'P',
            'Time Overall (s)': 'TOTAL',
            'Time Computation (s)': 'WORK',
            'Time Other (s)': 'OTHER'
        }
        
        for df in [omp_df, pthread_df, mpi_df]:
            df.rename(columns=column_mapping, inplace=True)
        
        # Add implementation type
        omp_df['RUN_TYPE'] = 'OMP'
        pthread_df['RUN_TYPE'] = 'PTH'
        mpi_df['RUN_TYPE'] = 'MPI'
        
        # Combine all data
        df = pd.concat([omp_df, pthread_df, mpi_df])
        
        # Add matrix size info
        df['COLS'] = df['ROWS']  # Since matrices are square
        df['matrix_size'] = df['ROWS'].astype(str) + 'x' + df['COLS'].astype(str)
        df['size_num'] = df['ROWS'].astype(int)
        
        # Calculate speedup and efficiency
        results = []
        for size in df['matrix_size'].unique():
            size_data = df[df['matrix_size'] == size].copy()
            for run_type in ['OMP', 'PTH', 'MPI']:
                type_data = size_data[size_data['RUN_TYPE'] == run_type].copy()
                if not type_data.empty:
                    serial_time = type_data[type_data['P'] == 1]['TOTAL'].iloc[0]
                    type_data['speedup'] = serial_time / type_data['TOTAL']
                    type_data['efficiency'] = type_data['speedup'] / type_data['P']
                    results.append(type_data)
        
        return pd.concat(results)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSV files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing the CSV files: {e}")
        sys.exit(1)

def setup_y_axis(ax, data_min, data_max, is_log=True):
    """Setup y-axis with appropriate tick marks and labels"""
    if is_log:
        log_min = np.floor(np.log10(data_min))
        log_max = np.ceil(np.log10(data_max))
        
        major_ticks = []
        minor_ticks = []
        
        for exp in range(int(log_min), int(log_max + 1)):
            major_ticks.append(10**exp)
            if exp < log_max:
                for i in range(2, 10):
                    minor_ticks.append(i * 10**exp)
        
        # Create locators with the tick values
        major_locator = matplotlib.ticker.FixedLocator(major_ticks)
        minor_locator = matplotlib.ticker.FixedLocator(minor_ticks)
        
        # Set the locators on the axis
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_time_ticks))
    else:
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_time_ticks))

def create_metric_plots(df, metric, ylabel, title_prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    # Define styles for each implementation
    impl_styles = {
        'OMP': ('blue', 'o', '-', 'OpenMP'),
        'PTH': ('green', 's', '--', 'Pthreads'),
        'MPI': ('red', '^', ':', 'MPI')
    }
    
    # Create a plot for each matrix size
    for size in sorted(df['matrix_size'].unique()):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()  # Get current axis
        
        size_data = df[df['matrix_size'] == size]
        
        max_parallel = max(size_data['P'])
        if metric == 'speedup':
            x_ideal = range(1, max_parallel + 1)
            plt.plot(x_ideal, x_ideal, 'k--', label='Ideal', alpha=0.7)
        elif metric == 'efficiency':
            plt.axhline(y=1, color='k', linestyle='--', label='Ideal', alpha=0.7)
        
        for impl, (color, marker, line_style, label) in impl_styles.items():
            impl_data = size_data[size_data['RUN_TYPE'] == impl].sort_values('P')
            if not impl_data.empty:
                plt.plot(impl_data['P'], impl_data[metric],
                        color=color,
                        marker=marker,
                        linestyle=line_style,
                        label=label,
                        linewidth=2,
                        markersize=8)
        
        if metric == 'TOTAL':
            plt.yscale('log')
            setup_y_axis(ax, impl_data[metric].min(), impl_data[metric].max(), True)
        else:
            setup_y_axis(ax, impl_data[metric].min(), impl_data[metric].max(), False)
        
        plt.title(f"{title_prefix} - Matrix Size {size}")
        plt.xlabel("Number of Threads/Processes")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(sorted(df['P'].unique()))
        
        if metric in ['speedup', 'efficiency']:
            plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric.lower()}_{size.split('x')[0]}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_component_time_plots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate plots for each matrix size
    for size in sorted(df['matrix_size'].unique()):
        plt.figure(figsize=(15, 8))
        
        size_data = df[df['matrix_size'] == size]
        parallel_counts = sorted(size_data['P'].unique())
        implementations = ['OMP', 'PTH', 'MPI']
        
        bar_width = 0.25
        
        for i, run_type in enumerate(implementations):
            impl_data = size_data[size_data['RUN_TYPE'] == run_type].sort_values('P')
            x = range(len(parallel_counts))
            
            offset = i * bar_width
            plt.bar([p + offset for p in x], 
                   impl_data['WORK'], 
                   bar_width,
                   label=f'{run_type} Work',
                   alpha=0.8)
            
            plt.bar([p + offset for p in x], 
                   impl_data['OTHER'],
                   bar_width,
                   bottom=impl_data['WORK'],
                   label=f'{run_type} Other',
                   alpha=0.8,
                   hatch='//')
        
        plt.xlabel('Number of Threads/Processes')
        plt.ylabel('Time (seconds)')
        plt.title(f'Time Components Analysis - Matrix Size {size}')
        plt.xticks([p + bar_width * 1.5 for p in x], parallel_counts)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_components_{size.split("x")[0]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def print_summary(df):
    print("\nPerformance Summary:")
    implementation_names = {'OMP': 'OpenMP', 'PTH': 'Pthreads', 'MPI': 'MPI'}
    
    for run_type in ['OMP', 'PTH', 'MPI']:
        print(f"\n{implementation_names[run_type]} Implementation:")
        for size in sorted(df['matrix_size'].unique()):
            impl_data = df[(df['RUN_TYPE'] == run_type) & (df['matrix_size'] == size)]
            if not impl_data.empty:
                best_time = impl_data['TOTAL'].min()
                best_parallel = impl_data.loc[impl_data['TOTAL'].idxmin(), 'P']
                best_speedup = impl_data['speedup'].max()
                
                print(f"\nMatrix Size: {size}")
                print(f"  Best time: {best_time:.3f} seconds (with {best_parallel} threads/processes)")
                print(f"  Max speedup: {best_speedup:.2f}x")

def create_size_comparison_plot(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define styles for each implementation
    implementations = {
        'OMP': ('blue', 'o', '-', 'OpenMP'),
        'PTH': ('green', 's', '--', 'Pthreads'),
        'MPI': ('red', '^', ':', 'MPI')
    }
    
    all_times = []
    
    # Plot single thread/process comparison
    for run_type, (color, marker, line_style, label) in implementations.items():
        single_data = df[(df['RUN_TYPE'] == run_type) & (df['P'] == 1)].sort_values('size_num')
        if not single_data.empty:
            ax1.plot(single_data['size_num'], single_data['TOTAL'],
                    marker=marker,
                    color=color,
                    linestyle=line_style,
                    label=label,
                    linewidth=2,
                    markersize=8)
            all_times.extend(single_data['TOTAL'].values)
    
    # Plot best parallel times comparison
    for run_type, (color, marker, line_style, label) in implementations.items():
        # Group by size and find minimum times
        best_times_df = df[df['RUN_TYPE'] == run_type].groupby('size_num')['TOTAL'].min().reset_index()
        best_times_df = best_times_df.sort_values('size_num')
        
        if not best_times_df.empty:
            ax2.plot(best_times_df['size_num'], best_times_df['TOTAL'],
                    marker=marker,
                    color=color,
                    linestyle=line_style,
                    label=label,
                    linewidth=2,
                    markersize=8)
            all_times.extend(best_times_df['TOTAL'].values)

    plt.close()

def main():
    args = parse_arguments()
    
    print("Reading data from input files...")
    df = prepare_data(args.omp, args.pthread, args.mpi)
    
    print(f"Generating plots in directory: {args.output}")
    
    # Create individual metric plots
    create_metric_plots(df, 'TOTAL', 'Execution Time (seconds)', 'Execution Time', args.output)
    create_metric_plots(df, 'speedup', 'Speedup (T₁/Tₚ)', 'Speedup', args.output)
    create_metric_plots(df, 'efficiency', 'Efficiency (Speedup/P)', 'Efficiency', args.output)
    
    # Create timing component plots
    create_component_time_plots(df, args.output)
    
    # Create comparison plots
    create_size_comparison_plot(df, args.output)
    
    print_summary(df)
    
    print(f"\nPlots have been saved in the '{args.output}' directory")

if __name__ == "__main__":
    main()