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
import sys
import time

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


# Progress tracking function
def print_progress(current, total, start_time, prefix="Progress"):
    """Print progress with time estimation."""
    elapsed = time.time() - start_time
    percent = 100.0 * current / total if total > 0 else 0

    # Estimate time remaining
    if current > 0:
        time_per_item = elapsed / current
        remaining_items = total - current
        eta = time_per_item * remaining_items
        eta_str = f", ETA: {eta:.1f}s" if eta < 120 else f", ETA: {eta / 60:.1f}m"
    else:
        eta_str = ""

    print(f"\r{prefix}: {current}/{total} ({percent:.1f}%){eta_str}", end="", flush=True)
    if current == total:
        print("", flush=True)  # Add newline at the end


# Add to run_backtest.py in the main function
def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run backtests with systematic validation steps.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy to backtest.')
    parser.add_argument('--data', type=str, help='Path to data file (overrides config).')
    parser.add_argument('--mes-data', action='store_true', help='Indicate that the data is MES futures format.')
    parser.add_argument('--steps', type=str, default='1,2,3,4',
                        help='Validation steps to run (comma-separated, e.g., 1,2,3,4).')
    parser.add_argument('--train_start', type=str, help='Start date for training period (YYYY-MM-DD).')
    parser.add_argument('--train_end', type=str, help='End date for training period (YYYY-MM-DD).')
    parser.add_argument('--jobs', type=int, default=-1,
                        help='Number of parallel jobs for permutation test. -1 for all cores.')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output.')
    args = parser.parse_args()

    quiet = args.quiet

    if not quiet:
        print("Starting backtest...")

    # Load configuration
    config = Config.load_yaml(args.config)
    # Override config values with command-line arguments if provided
    if args.train_start:
        config['validation']['insample']['train_start'] = args.train_start
    if args.train_end:
        config['validation']['insample']['train_end'] = args.train_end

    # Parse validation steps
    steps = [int(step) for step in args.steps.split(',')]

    # Get data filepath
    data_filepath = args.data if args.data else config['data']['filepath']

    # Load data
    data_loader = DataLoader()

    # Start timing
    start_time = time.time()

    # Check data type from config
    data_type = config['data'].get('data_type', 'standard')
    if args.mes_data or data_type == 'mes_futures' or 'U19_H25' in data_filepath:
        try:
            data = data_loader.load_mes_futures(data_filepath)
            if not quiet:
                print(f"Loaded MES futures data: {len(data)} rows")
        except AttributeError:
            # Fallback if method not available
            data = data_loader.load_csv(
                data_filepath,
                date_column='datetime',
                datetime_format=None
            )
            if not quiet:
                print(f"Loaded data with standard loader: {len(data)} rows")
    else:
        data = data_loader.load_csv(
            data_filepath,
            date_column=config['data']['date_column'],
            datetime_format=config['data']['datetime_format']
        )
        if not quiet:
            print(f"Loaded standard data: {len(data)} rows")

    # Get strategy configuration
    strategy_config = Config.get_strategy_config(config, args.strategy)

    # Create strategy instance
    strategy = StrategyFactory.create_strategy(args.strategy, strategy_config.get('default_params'))

    if not quiet:
        print(f"Created strategy: {strategy.name}")

    # Create backtester
    backtester = Backtester(strategy, data)

    # Get parameter grid
    parameter_grid = Config.get_parameter_grid(strategy_config)

    # Calculate total parameter combinations for progress tracking
    total_combinations = 1
    for param, values in parameter_grid.items():
        total_combinations *= len(values)

    if not quiet:
        print(f"Total parameter combinations to test: {total_combinations}")

    # Add a counter function to track progress
    current_combination = [0]  # Use a list for mutability

    # Define a tracking function to be called by the optimization process
    def track_progress():
        current_combination[0] += 1
        print_progress(current_combination[0], total_combinations, start_time, "Optimization progress")

    # Attach the tracking function to the strategy
    if hasattr(strategy, 'set_progress_callback'):
        strategy.set_progress_callback(track_progress)

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
        if not quiet:
            print("\n===== Step 1: In-Sample Optimization =====")
            print(f"Starting optimization with {total_combinations} parameter combinations...")

        opt_start_time = time.time()

        optimization_results = backtester.run_insample_optimization(
            train_start=config['validation']['insample']['train_start'],
            train_end=config['validation']['insample']['train_end'],
            parameter_grid=parameter_grid,
            objective_func=objective_func,
            n_jobs=args.jobs  # Add this line to pass the jobs parameter

        )

        opt_elapsed = time.time() - opt_start_time
        if not quiet:
            print(f"\nOptimization completed in {opt_elapsed:.2f} seconds")

        print(f"Best parameters: {optimization_results['best_params']}")
        print(f"Best {objective_func.__name__}: {optimization_results['best_value']:.4f}")

        # Plot in-sample performance
        if config['output']['plot_results']:
            backtester.plot_insample_performance()

        # Display correct trade count if available
        if hasattr(strategy, 'last_trade_count'):
            # Create corrected metrics
            corrected_metrics = backtester.metrics.copy()
            corrected_metrics['num_trades'] = float(strategy.last_trade_count)

            print("\n===== Trading Statistics =====")
            print(f"Total trades executed: {strategy.last_trade_count}")
            if hasattr(strategy, 'last_exits_by_max_bars'):
                print(f"Exits by max bars held: {strategy.last_exits_by_max_bars}")
            if hasattr(strategy, 'last_exits_by_stop_loss'):
                print(f"Exits by stop loss: {strategy.last_exits_by_stop_loss}")
            if hasattr(strategy, 'last_exits_by_trailing_stop'):
                print(f"Exits by trailing stop: {strategy.last_exits_by_trailing_stop}")

            print("\n===== Performance Metrics =====")
            for key, value in corrected_metrics.items():
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            # Fallback to original metrics if the strategy doesn't have the last_trade_count attribute
            print("\n===== Performance Metrics =====")
            for key, value in backtester.metrics.items():
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")

    if 2 in steps:
        if not quiet:
            print("\n===== Step 2: In-Sample Permutation Test =====")

        perm_start_time = time.time()
        n_permutations = config['validation']['insample_test']['n_permutations']
        print(f"Running {n_permutations} permutations...")

        try:
            test_results = backtester.run_insample_permutation_test(
                train_start=config['validation']['insample']['train_start'],
                train_end=config['validation']['insample']['train_end'],
                parameter_grid=parameter_grid,
                objective_func=objective_func,
                n_permutations=n_permutations,
                show_plot=False,
                n_jobs=args.jobs
            )

            perm_elapsed = time.time() - perm_start_time
            if not quiet:
                print(f"\nPermutation test completed in {perm_elapsed:.2f} seconds")

            print(f"P-value: {test_results['p_value']:.4f}")
            print(f"Real {objective_func.__name__}: {test_results['real_objective']:.4f}")

            # Decision based on p-value
            threshold = 0.01
            if test_results['p_value'] <= threshold:
                print(f"PASS: p-value ({test_results['p_value']:.4f}) <= threshold ({threshold})")
            else:
                print(f"FAIL: p-value ({test_results['p_value']:.4f}) > threshold ({threshold})")
                if 3 not in steps and 4 not in steps:
                    print("Stopping due to failed in-sample permutation test.")
                    return

        except Exception as e:
            print(f"ERROR in permutation test: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing with other steps...")

    if 3 in steps:
        if not quiet:
            print("\n===== Step 3: Walk-Forward Optimization =====")

        wf_start_time = time.time()

        walkforward_results = backtester.run_walkforward_optimization(
            train_window=config['validation']['walkforward']['train_window'],
            train_interval=config['validation']['walkforward']['train_interval'],
            parameter_grid=parameter_grid,
            objective_func=objective_func,
            show_optimization_progress=config['validation']['walkforward']['show_optimization_progress']
        )

        wf_elapsed = time.time() - wf_start_time
        if not quiet:
            print(f"\nWalk-forward optimization completed in {wf_elapsed:.2f} seconds")

        # Print metrics
        metrics = walkforward_results['metrics']

        # Display correct trade count if available for walk-forward test
        if hasattr(strategy, 'last_trade_count'):
            # Create corrected walk-forward metrics
            corrected_metrics = metrics.copy()
            corrected_metrics['num_trades'] = float(strategy.last_trade_count)

            print("\n===== Walk-Forward Trading Statistics =====")
            print(f"Total trades executed: {strategy.last_trade_count}")
            if hasattr(strategy, 'last_exits_by_max_bars'):
                print(f"Exits by max bars held: {strategy.last_exits_by_max_bars}")
            if hasattr(strategy, 'last_exits_by_stop_loss'):
                print(f"Exits by stop loss: {strategy.last_exits_by_stop_loss}")
            if hasattr(strategy, 'last_exits_by_trailing_stop'):
                print(f"Exits by trailing stop: {strategy.last_exits_by_trailing_stop}")

            print("\n===== Walk-Forward Performance Metrics =====")
            for key, value in corrected_metrics.items():
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            # Fallback to original metrics
            print("\n===== Walk-Forward Performance Metrics =====")
            print(f"Walk-forward {objective_func.__name__}: {metrics[objective_func.__name__]:.4f}")
            print(f"Walk-forward total return: {metrics['total_return']:.4f}")
            print(f"Walk-forward Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"Walk-forward max drawdown: {metrics['max_drawdown']:.4f}")

        # Plot walk-forward performance
        if config['output']['plot_results']:
            backtester.plot_walkforward_performance()

    if 4 in steps:
        if not quiet:
            print("\n===== Step 4: Walk-Forward Permutation Test =====")

        wf_perm_start_time = time.time()
        wf_n_permutations = config['validation']['walkforward_test']['n_permutations']
        print(f"Running {wf_n_permutations} walk-forward permutations...")

        wf_test_results = backtester.run_walkforward_permutation_test(
            train_data_start=config['validation']['walkforward_test']['train_data_start'],
            train_data_end=config['validation']['walkforward_test']['train_data_end'],
            test_data_start=config['validation']['walkforward_test']['test_data_start'],
            test_data_end=config['validation']['walkforward_test']['test_data_end'],
            parameter_grid=parameter_grid,
            objective_func=objective_func,
            n_permutations=wf_n_permutations,
            show_plot=config['validation']['walkforward_test']['show_plot'],
            n_jobs=args.jobs
        )

        wf_perm_elapsed = time.time() - wf_perm_start_time
        if not quiet:
            print(f"\nWalk-forward permutation test completed in {wf_perm_elapsed:.2f} seconds")

        print(f"Walk-forward p-value: {wf_test_results['p_value']:.4f}")
        print(f"Walk-forward real {objective_func.__name__}: {wf_test_results['real_objective']:.4f}")

        # Decision based on p-value
        threshold = 0.05
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

        # Update metrics in backtester before saving if we have correct trade count
        if hasattr(strategy, 'last_trade_count'):
            backtester.metrics['num_trades'] = float(strategy.last_trade_count)

        backtester.save_results(results_filepath)
        print(f"Results saved to: {results_filepath}")

    # Print total elapsed time
    total_elapsed = time.time() - start_time
    if total_elapsed < 60:
        print(f"\nTotal execution time: {total_elapsed:.2f} seconds")
    else:
        minutes = int(total_elapsed // 60)
        seconds = total_elapsed % 60
        print(f"\nTotal execution time: {minutes} minutes {seconds:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)