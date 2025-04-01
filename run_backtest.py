#!/usr/bin/env python3
"""
Main script for running backtests with the four validation steps:
1. In-sample optimization
2. In-sample permutation test
3. Walk-forward test
4. Walk-forward permutation test
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pprint import pprint

from utils.config import Config
from utils.data_loader import DataLoader
from utils.metrics import profit_factor, sharpe_ratio, drawdown, cagr, calmar_ratio, sortino_ratio
from engine.strategy_factory import StrategyFactory
from engine.backtester import Backtester

# Dictionary of available objective functions
OBJECTIVE_FUNCTIONS = {
    'profit_factor': profit_factor,
    'sharpe_ratio': sharpe_ratio,
    'drawdown': drawdown,
    'cagr': cagr,
    'calmar_ratio': calmar_ratio,
    'sortino_ratio': sortino_ratio
}


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run backtests with systematic validation steps.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy to backtest.')
    parser.add_argument('--data', type=str, help='Path to data file (overrides config).')
    parser.add_argument('--steps', type=str, default='1,2,3,4', help='Validation steps to run (comma-separated, e.g., 1,2,3,4).')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load_yaml(args.config)
    
    # Parse validation steps
    steps = [int(step) for step in args.steps.split(',')]
    
    # Get data filepath
    data_filepath = args.data if args.data else config['data']['filepath']
    
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_csv(
        data_filepath,
        date_column=config['data']['date_column'],
        datetime_format=config['data']['datetime_format']
    )
    
    print(f"Loaded data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    # Get strategy configuration
    strategy_config = Config.get_strategy_config(config, args.strategy)
    
    # Create strategy instance
    strategy = StrategyFactory.create_strategy(args.strategy, strategy_config.get('default_params'))
    
    print(f"Created strategy: {strategy.name}")
    
    # Create backtester
    backtester = Backtester(strategy, data)
    
    # Get parameter grid
    parameter_grid = Config.get_parameter_grid(strategy_config)
    
    # Get objective function
    objective_func_name = config['validation']['insample']['objective_function']
    objective_func = OBJECTIVE_FUNCTIONS[objective_func_name]
    
    # Create results directory
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run validation steps
    if 1 in steps:
        print("\n===== Step 1: In-Sample Optimization =====")
        optimization_results = backtester.run_insample_optimization(
            train_start=config['validation']['insample']['train_start'],
            train_end=config['validation']['insample']['train_end'],
            parameter_grid=parameter_grid,
            objective_func=objective_func
        )
        
        print(f"Best parameters: {optimization_results['best_params']}")
        print(f"Best {objective_func.__name__}: {optimization_results['best_value']:.4f}")
        
        # Plot in-sample performance
        if config['output']['plot_results']:
            backtester.plot_insample_performance()
    
    if 2 in steps:
        print("\n===== Step 2: In-Sample Permutation Test =====")
        test_results = backtester.run_insample_permutation_test(
            train_start=config['validation']['insample']['train_start'],
            train_end=config['validation']['insample']['train_end'],
            parameter_grid=parameter_grid,
            objective_func=objective_func,
            n_permutations=config['validation']['insample_test']['n_permutations'],
            show_plot=config['validation']['insample_test']['show_plot']
        )
        
        print(f"P-value: {test_results['p_value']:.4f}")
        print(f"Real {objective_func.__name__}: {test_results['real_objective']:.4f}")
        
        # Decision based on p-value
        threshold = 0.01  # As mentioned in the video
        if test_results['p_value'] <= threshold:
            print(f"PASS: p-value ({test_results['p_value']:.4f}) <= threshold ({threshold})")
        else:
            print(f"FAIL: p-value ({test_results['p_value']:.4f}) > threshold ({threshold})")
            if 3 not in steps and 4 not in steps:
                print("Stopping due to failed in-sample permutation test.")
                return
    
    if 3 in steps:
        print("\n===== Step 3: Walk-Forward Optimization =====")
        walkforward_results = backtester.run_walkforward_optimization(
            train_window=config['validation']['walkforward']['train_window'],
            train_interval=config['validation']['walkforward']['train_interval'],
            parameter_grid=parameter_grid,
            objective_func=objective_func,
            show_optimization_progress=config['validation']['walkforward']['show_optimization_progress']
        )
        
        # Print metrics
        metrics = walkforward_results['metrics']
        print(f"Walk-forward {objective_func.__name__}: {metrics[objective_func.__name__]:.4f}")
        print(f"Walk-forward total return: {metrics['total_return']:.4f}")
        print(f"Walk-forward Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Walk-forward max drawdown: {metrics['max_drawdown']:.4f}")
        
        # Plot walk-forward performance
        if config['output']['plot_results']:
            backtester.plot_walkforward_performance()
    
    if 4 in steps:
        print("\n===== Step 4: Walk-Forward Permutation Test =====")
        wf_test_results = backtester.run_walkforward_permutation_test(
            train_data_start=config['validation']['walkforward_test']['train_data_start'],
            train_data_end=config['validation']['walkforward_test']['train_data_end'],
            test_data_start=config['validation']['walkforward_test']['test_data_start'],
            test_data_end=config['validation']['walkforward_test']['test_data_end'],
            parameter_grid=parameter_grid,
            objective_func=objective_func,
            n_permutations=config['validation']['walkforward_test']['n_permutations'],
            show_plot=config['validation']['walkforward_test']['show_plot']
        )
        
        print(f"Walk-forward p-value: {wf_test_results['p_value']:.4f}")
        print(f"Walk-forward real {objective_func.__name__}: {wf_test_results['real_objective']:.4f}")
        
        # Decision based on p-value
        threshold = 0.05  # A bit more lenient as mentioned in the video
        if wf_test_results['p_value'] <= threshold:
            print(f"PASS: walk-forward p-value ({wf_test_results['p_value']:.4f}) <= threshold ({threshold})")
            print("Strategy is validated and can be considered for live trading.")
        else:
            print(f"FAIL: walk-forward p-value ({wf_test_results['p_value']:.4f}) > threshold ({threshold})")
            print("Strategy failed to pass walk-forward permutation test. Not recommended for live trading.")
    
    # Save results
    if config['output']['save_results']:
        results_filepath = os.path.join(
            results_dir,
            f"{args.strategy}_{timestamp}.json"
        )
        backtester.save_results(results_filepath)


if __name__ == "__main__":
    main()
