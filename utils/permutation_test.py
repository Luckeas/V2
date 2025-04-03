import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from joblib import Parallel, delayed


def permute_bars(data: pd.DataFrame, start_index: int = 0) -> pd.DataFrame:
    """
    Permute the bars of market data while preserving some statistical properties.
    Based on the approach described in the video.

    Args:
        data (pd.DataFrame): Market data with OHLC prices.
        start_index (int, optional): Starting index for permutation. Defaults to 0.

    Returns:
        pd.DataFrame: Permuted market data.
    """
    # Convert to numpy array for faster operations
    prices = data[['open', 'high', 'low', 'close']].values

    # Take log of prices
    log_prices = np.log(prices)

    # Calculate relative prices and gaps
    n_bars = len(log_prices)
    rel_prices = np.zeros((n_bars - start_index, 3))  # high, low, close relative to open
    rel_opens = np.zeros(n_bars - start_index)  # open relative to previous close (gap)

    # Store first bar (will remain unchanged)
    first_bar = log_prices[start_index].copy()

    # Calculate relative prices
    for i in range(start_index, n_bars):
        bar_idx = i - start_index
        # Relative high, low, close to open
        rel_prices[bar_idx, 0] = log_prices[i, 1] - log_prices[i, 0]  # high - open
        rel_prices[bar_idx, 1] = log_prices[i, 2] - log_prices[i, 0]  # low - open
        rel_prices[bar_idx, 2] = log_prices[i, 3] - log_prices[i, 0]  # close - open

        # Open gap (except for the first bar)
        if i > start_index:
            rel_opens[bar_idx] = log_prices[i, 0] - log_prices[i - 1, 3]  # open - prev close

    # Shuffle the relative prices and gaps
    np.random.shuffle(rel_prices)
    np.random.shuffle(rel_opens)

    # Create permuted prices
    permuted_log_prices = np.zeros_like(log_prices)

    # Copy unchanged prices before start_index
    if start_index > 0:
        permuted_log_prices[:start_index] = log_prices[:start_index]

    # Set first bar in permutation
    permuted_log_prices[start_index] = first_bar

    # String together the permuted prices
    for i in range(start_index + 1, n_bars):
        bar_idx = i - start_index
        prev_idx = i - 1

        # Set open based on previous close and gap
        permuted_log_prices[i, 0] = permuted_log_prices[prev_idx, 3] + rel_opens[bar_idx]

        # Set high, low, close based on open and relative prices
        permuted_log_prices[i, 1] = permuted_log_prices[i, 0] + rel_prices[bar_idx, 0]  # high
        permuted_log_prices[i, 2] = permuted_log_prices[i, 0] + rel_prices[bar_idx, 1]  # low
        permuted_log_prices[i, 3] = permuted_log_prices[i, 0] + rel_prices[bar_idx, 2]  # close

    # Convert back to normal scale
    permuted_prices = np.exp(permuted_log_prices)

    # Create DataFrame with permuted prices
    permuted_df = pd.DataFrame(
        permuted_prices,
        index=data.index,
        columns=['open', 'high', 'low', 'close']
    )

    # Add any additional columns from the original DataFrame
    for col in data.columns:
        if col not in ['open', 'high', 'low', 'close']:
            permuted_df[col] = data[col].values

    return permuted_df


def _run_single_permutation(i, strategy, data, objective_func, parameter_grid):
    """Run a single permutation and return the result."""
    try:
        # Generate permuted data
        permuted_data = permute_bars(data)

        # Optimize strategy on permuted data
        perm_params, perm_objective = strategy.optimize(permuted_data, objective_func, parameter_grid)

        return i, perm_objective, True, None
    except Exception as e:
        return i, None, False, str(e)


def insample_permutation_test(
        strategy,
        data: pd.DataFrame,
        objective_func: callable,
        parameter_grid: Dict[str, List[Any]],
        n_permutations: int = 1000,
        show_plot: bool = True,
        n_jobs: int = -1  # -1 uses all available cores
) -> Tuple[float, Dict[str, Any], float, List[float]]:
    """
    Perform in-sample permutation test as described in the video, with parallel processing.

    Args:
        strategy: Trading strategy instance.
        data (pd.DataFrame): Market data with OHLC prices.
        objective_func (callable): Function to evaluate performance.
        parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.
        n_permutations (int, optional): Number of permutations. Defaults to 1000.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
        n_jobs (int, optional): Number of jobs for parallel processing. -1 for all cores. Defaults to -1.

    Returns:
        Tuple[float, Dict[str, Any], float, List[float]]:
            P-value, best parameters, real objective value, permutation objective values.
    """
    print("Starting in-sample permutation test function with parallel processing...", flush=True)
    print(f"Optimizing strategy on real data", flush=True)

    # Optimize strategy on real data
    best_params, real_objective = strategy.optimize(data, objective_func, parameter_grid)

    print(f"Real data optimization complete. Best params: {best_params}", flush=True)
    print(f"Real {objective_func.__name__}: {real_objective:.4f}", flush=True)

    # Keep track of permutation objective values
    perm_objectives = []
    count_better = 0

    # Run permutation tests in parallel
    print(f"Running {n_permutations} permutations in parallel with {n_jobs} jobs...", flush=True)

    # Execute permutations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_permutation)(i, strategy, data, objective_func, parameter_grid)
        for i in range(n_permutations)
    )

    # Process results
    for i, perm_objective, success, error_msg in results:
        if success:
            # Check if permutation result is better than or equal to real result
            if objective_func.__name__ == "drawdown":
                is_better = perm_objective <= real_objective
            else:
                is_better = perm_objective >= real_objective

            if is_better:
                count_better += 1

            perm_objectives.append(perm_objective)
        else:
            print(f"Error in permutation {i}: {error_msg}", flush=True)

    # Calculate p-value
    if len(perm_objectives) > 0:
        p_value = count_better / len(perm_objectives)
    else:
        p_value = 1.0  # Default to worst case if all permutations failed

    print(f"Permutation test completed.", flush=True)
    print(f"P-value: {p_value:.4f}", flush=True)
    print(f"Count of better permutations: {count_better} out of {len(perm_objectives)}", flush=True)

    # Plot histogram of permutation results
    if show_plot and len(perm_objectives) > 0:
        try:
            print("About to start histogram plot creation...", flush=True)
            plt.figure(figsize=(10, 6))
            plt.hist(perm_objectives, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(real_objective, color='red', linestyle='dashed', linewidth=2)
            plt.title(f'In-Sample Permutation Test Results (P-value: {p_value:.4f})')
            plt.xlabel(f'{objective_func.__name__}')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()
            print("Histogram plot displayed", flush=True)
        except Exception as e:
            print(f"Error creating histogram: {str(e)}", flush=True)

    print("Returning permutation test results...", flush=True)
    return p_value, best_params, real_objective, perm_objectives


def walkforward_permutation_test(
        strategy,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        objective_func: callable,
        parameter_grid: Dict[str, List[Any]],
        n_permutations: int = 200,
        show_plot: bool = True,
        n_jobs: int = -1
) -> Tuple[float, Dict[str, Any], float, List[float]]:
    """
    Perform walk-forward permutation test as described in the video.

    Args:
        strategy: Trading strategy instance.
        train_data (pd.DataFrame): Training data for optimization.
        test_data (pd.DataFrame): Test data for out-of-sample evaluation.
        objective_func (callable): Function to evaluate performance.
        parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.
        n_permutations (int, optional): Number of permutations. Defaults to 200.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1.

    Returns:
        Tuple[float, Dict[str, Any], float, List[float]]:
            P-value, best parameters, real objective value, permutation objective values.
    """
    print("Starting walk-forward permutation test with parallel processing...", flush=True)

    # Optimize strategy on training data
    print("Optimizing strategy on training data...", flush=True)
    best_params, _ = strategy.optimize(train_data, objective_func, parameter_grid)
    print(f"Best parameters from training: {best_params}", flush=True)

    # Generate signals for test data using optimized parameters
    signals = strategy.generate_signals(test_data, best_params)

    # Calculate returns on test data
    returns = strategy.compute_returns(test_data, signals)

    # Calculate real objective value on test data
    real_objective = objective_func(returns)
    print(f"Real {objective_func.__name__} on test data: {real_objective:.4f}", flush=True)

    # Keep track of permutation objective values
    perm_objectives = []
    count_better = 0

    # Concatenate train and test data for reference
    combined_data = pd.concat([train_data, test_data])
    train_size = len(train_data)

    # Helper function for parallel execution
    def _run_walkforward_permutation(i):
        try:
            # Generate permuted test data (permutation starts after training data)
            permuted_data = permute_bars(combined_data, start_index=train_size)
            permuted_test_data = permuted_data.iloc[train_size:]

            # Generate signals for permuted test data using same optimized parameters
            perm_signals = strategy.generate_signals(permuted_test_data, best_params)

            # Calculate returns on permuted test data
            perm_returns = strategy.compute_returns(permuted_test_data, perm_signals)

            # Calculate objective value on permuted test data
            perm_objective = objective_func(perm_returns)

            return i, perm_objective, True, None
        except Exception as e:
            return i, None, False, str(e)

    # Run permutation tests in parallel
    print(f"Running {n_permutations} walk-forward permutations in parallel with {n_jobs} jobs...", flush=True)

    # Execute permutations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_walkforward_permutation)(i) for i in range(n_permutations)
    )

    # Process results
    for i, perm_objective, success, error_msg in results:
        if success:
            # Check if permutation result is better than or equal to real result
            if objective_func.__name__ == "drawdown":
                is_better = perm_objective <= real_objective
            else:
                is_better = perm_objective >= real_objective

            if is_better:
                count_better += 1

            perm_objectives.append(perm_objective)
        else:
            print(f"Error in permutation {i}: {error_msg}", flush=True)

    # Calculate p-value
    if len(perm_objectives) > 0:
        p_value = count_better / len(perm_objectives)
    else:
        p_value = 1.0  # Default to worst case if all permutations failed

    print(f"Walk-forward permutation test completed.", flush=True)
    print(f"P-value: {p_value:.4f}", flush=True)
    print(f"Count of better permutations: {count_better} out of {len(perm_objectives)}", flush=True)

    # Plot histogram of permutation results
    if show_plot and len(perm_objectives) > 0:
        try:
            print("About to start histogram plot creation...", flush=True)
            plt.figure(figsize=(10, 6))
            plt.hist(perm_objectives, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(real_objective, color='red', linestyle='dashed', linewidth=2)
            plt.title(f'Walk-Forward Permutation Test Results (P-value: {p_value:.4f})')
            plt.xlabel(f'{objective_func.__name__}')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()
            print("Histogram plot displayed", flush=True)
        except Exception as e:
            print(f"Error creating histogram: {str(e)}", flush=True)

    return p_value, best_params, real_objective, perm_objectives