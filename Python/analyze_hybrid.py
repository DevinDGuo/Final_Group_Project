import pandas as pd
import numpy as np
import argparse
import sys
import os

def analyze_performance(df):
    """Analyze performance metrics for each matrix size and find best overall configuration"""
    matrix_sizes = sorted(df['Matrix Size'].unique())
    results = []
    
    # Analyze each matrix size
    for matrix_size in matrix_sizes:
        # Calculate baseline (serial) time
        baseline = df[(df['Matrix Size'] == matrix_size) & 
                     (df['Processes'] == 1) & 
                     (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
        
        # Calculate metrics for all configurations
        matrix_data = df[df['Matrix Size'] == matrix_size].copy()
        matrix_data['Speedup'] = baseline / matrix_data['Time Overall (s)']
        matrix_data['Total Processors'] = matrix_data['Processes'] * matrix_data['OMP Threads']
        matrix_data['Efficiency'] = matrix_data['Speedup'] / matrix_data['Total Processors']
        matrix_data['Serial Fraction'] = (1/matrix_data['Speedup'] - 1/matrix_data['Total Processors']) / (1 - 1/matrix_data['Total Processors'])
        
        # Find best speedup configuration
        best_speedup_idx = matrix_data['Speedup'].idxmax()
        best_config = matrix_data.loc[best_speedup_idx]
        
        results.append({
            'Matrix Size': matrix_size,
            'Best Config Processes': int(best_config['Processes']),
            'Best Config Threads': int(best_config['OMP Threads']),
            'Total Processors': int(best_config['Total Processors']),
            'Speedup': best_config['Speedup'],
            'Efficiency': best_config['Efficiency'],
            'Serial Fraction': best_config['Serial Fraction'],
            'Time (s)': best_config['Time Overall (s)'],
            'Baseline Time (s)': baseline
        })
    
    # Find best overall configuration
    process_thread_scores = {}
    for process in df['Processes'].unique():
        for thread in df['OMP Threads'].unique():
            speedups = []
            for matrix_size in matrix_sizes:
                baseline = df[(df['Matrix Size'] == matrix_size) & 
                            (df['Processes'] == 1) & 
                            (df['OMP Threads'] == 1)]['Time Overall (s)'].iloc[0]
                try:
                    current_time = df[(df['Matrix Size'] == matrix_size) & 
                                    (df['Processes'] == process) & 
                                    (df['OMP Threads'] == thread)]['Time Overall (s)'].iloc[0]
                    speedups.append(baseline / current_time)
                except IndexError:
                    continue
            if speedups:
                process_thread_scores[(process, thread)] = np.mean(speedups)
    
    best_config = max(process_thread_scores.items(), key=lambda x: x[1])
    overall_best = {
        'Processes': int(best_config[0][0]),
        'Threads': int(best_config[0][1]),
        'Average Speedup': best_config[1]
    }
    
    return results, overall_best

def print_analysis_results(results, overall_best):
    """Format and print the analysis results"""
    print("\nPerformance Analysis Results:")
    print("=" * 80)
    
    for result in results:
        print(f"\nMatrix Size: {result['Matrix Size']}")
        print("-" * 40)
        print(f"Best Configuration:")
        print(f"  Processes: {result['Best Config Processes']}")
        print(f"  Threads: {result['Best Config Threads']}")
        print(f"  Total Processors: {result['Total Processors']}")
        print(f"\nPerformance Metrics:")
        print(f"  Baseline Time: {result['Baseline Time (s)']:.2f} seconds")
        print(f"  Best Time: {result['Time (s)']:.2f} seconds")
        print(f"  Speedup: {result['Speedup']:.2f}x")
        print(f"  Efficiency: {result['Efficiency']:.2f}")
        print(f"  Serial Fraction: {result['Serial Fraction']:.2f}")
    
    print("\nBest Overall Configuration:")
    print("=" * 80)
    print(f"Processes: {overall_best['Processes']}")
    print(f"Threads: {overall_best['Threads']}")
    print(f"Average Speedup across all matrix sizes: {overall_best['Average Speedup']:.2f}x")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze performance data from CSV file.",
        usage=f"python3 {os.path.basename(__file__)} <data.csv>"
    )
    parser.add_argument("filename", type=str, help="Path to the CSV file containing data.")
    return parser

def main():
    # Parse command line arguments
    parser = parse_arguments()
    args = parser.parse_args()

    # Read and validate input file
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
    
    # Perform analysis
    results, overall_best = analyze_performance(df)
    
    # Print results
    print_analysis_results(results, overall_best)

if __name__ == "__main__":
    main()