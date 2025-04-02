#!/usr/bin/env python3
"""
Simplified test script for Morning Reversal strategy optimization.
This script allows quick testing with a reduced parameter space.
"""

import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path to import from main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from utils.metrics import profit_factor, sharpe_ratio, calmar_ratio
from strategies.morning_reversal import MorningReversal


def main():
    """Run a simplified optimization test for Morning Reversal strategy."""
    print("Morning Reversal Strategy Test")
    print("-" * 50)
    
    # Load configuration
    config_path = 'reduced_config.yaml'  # Path to your reduced config
    if not os.path.exists(config_path):
        # If reduced config doesn't exist, use default
        config_path = 'config.yaml'
        print(f"Reduced config not found, using default: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get strategy configuration
    strategy_config = config.get('strategies', {}).get('morning_reversal', {})
    default_params = strategy_config.get('default_params', {})

    # Use reduced parameter grid if available, otherwise use full parameter grid
    if 'reduced_parameter_grid' in strategy_config:
        print("Using reduced parameter grid from config")
        parameter_grid_config = strategy_config.get('reduced_parameter_grid', {})
    else:
        print("Using full parameter grid from config")
        parameter_grid_config = strategy_config.get('parameter_grid', {})

    # Convert to the format expected by the strategy
    parameter_grid = {}
    for param_name, param_config in parameter_grid_config.items():
        if 'values' in param_config:
            parameter_grid[param_name] = param_config['values']
        elif 'range' in param_config:
            start = param_config['range']['start']
            end = param_config['range']['end']
            step = param_config['range'].get('step', 1)
            parameter_grid[param_name] = list(range(start, end + 1, step))
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    data_path = config.get('data', {}).get('filepath', '/Users/martinshih/Downloads/Systematic/Candlestick_Data/MES_Data/U19_H25.csv')
    
    try:
        # First try loading as MES futures data
        data = data_loader.load_mes_futures(data_path)
        print(f"Loaded MES futures data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        # Fallback to standard CSV loader
        print(f"Error loading MES data: {e}")
        print("Falling back to standard CSV loader...")
        data = data_loader.load_csv(data_path)
        print(f"Loaded data: {len(data)} rows")
    
    # Filter data for 2023 if needed
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    print(f"Filtering data for period: {start_date} to {end_date}")
    
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
    print(f"Filtered data: {len(filtered_data)} rows from {filtered_data.index[0]} to {filtered_data.index[-1]}")
    
    # Create strategy instance
    strategy = MorningReversal(**default_params)
    print(f"Created strategy: {strategy.name}")
    
    # Define objective function to maximize
    objective_func = profit_factor
    print(f"Objective function: {objective_func.__name__}")
    
    # Run optimization
    print("\nRunning optimization...")
    start_time = time.time()
    
    best_params, best_value = strategy.optimize(filtered_data, objective_func, parameter_grid)
    
    total_time = time.time() - start_time
    
    # Print results
    print("\nOptimization Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Best {objective_func.__name__}: {best_value:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Generate signals with optimized parameters
    print("\nGenerating signals with optimized parameters...")
    signals = strategy.generate_signals(filtered_data, best_params)
    signal_count = np.sum(signals != 0)
    print(f"Generated {signal_count} trading signals")
    
    # Calculate returns
    returns = strategy.compute_returns(filtered_data, signals)
    
    # Calculate performance metrics
    metrics = {
        'profit_factor': profit_factor(returns),
        'sharpe_ratio': sharpe_ratio(returns),
        'calmar_ratio': calmar_ratio(returns),
        'total_return': np.exp(np.sum(returns)) - 1,
        'win_rate': len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0,
        'num_trades': np.sum(np.abs(np.diff(signals, prepend=0)) > 0)
    }
    
    # Print metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"morning_reversal_{timestamp}.txt")
    
    with open(results_path, 'w') as f:
        f.write("Morning Reversal Strategy Optimization Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data period: {filtered_data.index[0]} to {filtered_data.index[-1]}\n")
        f.write(f"Optimization time: {total_time:.2f} seconds\n\n")
        
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        f.write(f"\nBest {objective_func.__name__}: {best_value:.4f}\n\n")
        
        f.write("Performance Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
