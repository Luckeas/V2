import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from utils.metrics import profit_factor, sharpe_ratio, drawdown, cagr, calmar_ratio
from utils.permutation_test import insample_permutation_test, walkforward_permutation_test
from engine.walk_forward import WalkForwardOptimizer


class Backtester:
    """
    Main backtesting engine implementing the four validation steps described in the video:
    1. In-sample excellence
    2. In-sample Monte Carlo permutation test
    3. Walk-forward test
    4. Walk-forward Monte Carlo permutation test
    """

    def __init__(self, strategy, data: pd.DataFrame = None):
        """
        Initialize the backtester.

        Args:
            strategy: Strategy instance to backtest.
            data (pd.DataFrame, optional): Market data with OHLC prices. Defaults to None.
        """
        self.strategy = strategy
        self.data = data
        self.signals = None
        self.returns = None
        self.metrics = {}
        self.optimization_results = {}
        self.insample_test_results = {}
        self.walkforward_results = {}
        self.walkforward_test_results = {}

    def set_data(self, data: pd.DataFrame):
        """
        Set the data for backtesting.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
        """
        self.data = data

    def run_insample_backtest(self,
                              train_start: Union[str, int] = None,
                              train_end: Union[str, int] = None,
                              parameters: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Run in-sample backtest.

        Args:
            train_start (Union[str, int], optional): Start of training period. Defaults to None.
            train_end (Union[str, int], optional): End of training period. Defaults to None.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary with performance metrics.
        """
        if self.data is None:
            raise ValueError("Data not set. Use set_data() to set the data.")

        # Get training data
        if train_start is not None and train_end is not None:
            if isinstance(train_start, str) and isinstance(train_end, str):
                train_data = self.data.loc[train_start:train_end]
            else:
                train_data = self.data.iloc[train_start:train_end]
        else:
            train_data = self.data

        # Generate signals
        self.signals = self.strategy.generate_signals(train_data, parameters)

        # Calculate returns
        self.returns, active_positions = self.strategy.compute_returns(train_data, self.signals)

        # Calculate metrics
        self.metrics = {
            'profit_factor': profit_factor(self.returns),
            'sharpe_ratio': sharpe_ratio(self.returns),
            'max_drawdown': drawdown(self.returns),
            'cagr': cagr(self.returns),
            'calmar_ratio': calmar_ratio(self.returns),
            'total_return': np.exp(np.sum(self.returns)) - 1,
            'win_rate': len(self.returns[self.returns > 0]) / len(self.returns[self.returns != 0]) if len(
                self.returns[self.returns != 0]) > 0 else 0,
            'num_trades': np.sum(np.diff(active_positions, prepend=0) != 0)  # Count actual position changes
        }

        return self.metrics

    def run_insample_optimization(self,
                                  train_start: Union[str, int] = None,
                                  train_end: Union[str, int] = None,
                                  parameter_grid: Dict[str, List[Any]] = None,
                                  objective_func: Callable = profit_factor,
                                  n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run in-sample optimization (Step 1).

        Args:
            train_start (Union[str, int], optional): Start of training period. Defaults to None.
            train_end (Union[str, int], optional): End of training period. Defaults to None.
            parameter_grid (Dict[str, List[Any]], optional): Grid of parameters to search. Defaults to None.
            objective_func (Callable, optional): Function to maximize during optimization. Defaults to profit_factor.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1 (all cores).

        Returns:
            Dict[str, Any]: Dictionary with optimization results.
        """
        if self.data is None:
            raise ValueError("Data not set. Use set_data() to set the data.")

        # Get training data
        if train_start is not None and train_end is not None:
            if isinstance(train_start, str) and isinstance(train_end, str):
                train_data = self.data.loc[train_start:train_end]
            else:
                train_data = self.data.iloc[train_start:train_end]
        else:
            train_data = self.data

        # Optimize strategy with n_jobs parameter
        best_params, best_value = self.strategy.optimize(train_data, objective_func, parameter_grid, n_jobs=n_jobs)

        # Store optimization results
        self.optimization_results = {
            'best_params': best_params,
            'best_value': best_value,
            'objective_func': objective_func.__name__
        }

        # Generate signals with optimized parameters
        self.signals = self.strategy.generate_signals(train_data, best_params)

        # Calculate returns
        self.returns = self.strategy.compute_returns(train_data, self.signals)

        # Calculate metrics
        self.metrics = {
            'profit_factor': profit_factor(self.returns),
            'sharpe_ratio': sharpe_ratio(self.returns),
            'max_drawdown': drawdown(self.returns),
            'cagr': cagr(self.returns),
            'calmar_ratio': calmar_ratio(self.returns),
            'total_return': np.exp(np.sum(self.returns)) - 1,
            'win_rate': len(self.returns[self.returns > 0]) / len(self.returns[self.returns != 0]) if len(
                self.returns[self.returns != 0]) > 0 else 0,
            'num_trades': self.strategy.trade_count if hasattr(self.strategy, 'trade_count') else np.sum(
                np.abs(np.diff(self.signals, prepend=0)) > 0)
        }

        return self.optimization_results

    def run_walkforward_optimization(self,
                                     train_window: Union[int, str] = "4Y",
                                     train_interval: Union[int, str] = "30D",
                                     parameter_grid: Dict[str, List[Any]] = None,
                                     objective_func: Callable = profit_factor,
                                     show_optimization_progress: bool = True) -> Dict[str, Any]:
        """
        Run walk-forward optimization (Step 3).

        Args:
            train_window (Union[int, str], optional): Training window (bars or time offset). Defaults to "4Y".
            train_interval (Union[int, str], optional): Retraining interval (bars or time offset). Defaults to "30D".
            parameter_grid (Dict[str, List[Any]], optional): Grid of parameters to search. Defaults to None.
            objective_func (Callable, optional): Function to maximize during optimization. Defaults to profit_factor.
            show_optimization_progress (bool, optional): Whether to show optimization progress. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary with walk-forward results.
        """
        if self.data is None:
            raise ValueError("Data not set. Use set_data() to set the data.")

        # Create walk-forward optimizer
        wf_optimizer = WalkForwardOptimizer(
            self.strategy, objective_func, parameter_grid, train_window, train_interval
        )

        # Run walk-forward optimization
        signals, results_df = wf_optimizer.run(self.data, show_optimization_progress)

        # Calculate returns
        returns = self.strategy.compute_returns(self.data, signals)

        # Calculate metrics
        metrics = {
            'profit_factor': profit_factor(returns),
            'sharpe_ratio': sharpe_ratio(returns),
            'max_drawdown': drawdown(returns),
            'cagr': cagr(returns),
            'calmar_ratio': calmar_ratio(returns),
            'total_return': np.exp(np.sum(returns)) - 1,
            'win_rate': len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0,
            'num_trades': self.strategy.trade_count if hasattr(self.strategy, 'trade_count') else np.sum(
                np.abs(np.diff(self.signals, prepend=0)) > 0)
        }

        # Store walk-forward results
        self.walkforward_results = {
            'signals': signals,
            'returns': returns,
            'metrics': metrics,
            'optimization_params': {
                'train_window': train_window,
                'train_interval': train_interval,
                'objective_func': objective_func.__name__
            },
            'results_df': results_df,
            'performance_metrics': wf_optimizer.performance_metrics
        }

        # Plot walk-forward results
        if show_optimization_progress:
            wf_optimizer.plot_optimization_results(self.data, signals)

        return self.walkforward_results

    def run_walkforward_permutation_test(self,
                                         train_data_start: Union[str, int] = None,
                                         train_data_end: Union[str, int] = None,
                                         test_data_start: Union[str, int] = None,
                                         test_data_end: Union[str, int] = None,
                                         parameter_grid: Dict[str, List[Any]] = None,
                                         objective_func: Callable = profit_factor,
                                         n_permutations: int = 200,
                                         show_plot: bool = True,
                                         n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run walk-forward Monte Carlo permutation test (Step 4).

        Args:
            train_data_start (Union[str, int], optional): Start of training period. Defaults to None.
            train_data_end (Union[str, int], optional): End of training period. Defaults to None.
            test_data_start (Union[str, int], optional): Start of test period. Defaults to None.
            test_data_end (Union[str, int], optional): End of test period. Defaults to None.
            parameter_grid (Dict[str, List[Any]], optional): Grid of parameters to search. Defaults to None.
            objective_func (Callable, optional): Function to maximize during optimization. Defaults to profit_factor.
            n_permutations (int, optional): Number of permutations. Defaults to 200.
            show_plot (bool, optional): Whether to show the plot. Defaults to True.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1.

        Returns:
            Dict[str, Any]: Dictionary with test results.
        """
        if self.data is None:
            raise ValueError("Data not set. Use set_data() to set the data.")

        # Get training and test data
        if isinstance(train_data_start, str) and isinstance(train_data_end, str):
            train_data = self.data.loc[train_data_start:train_data_end]
        else:
            train_data = self.data.iloc[train_data_start:train_data_end]

        if isinstance(test_data_start, str) and isinstance(test_data_end, str):
            test_data = self.data.loc[test_data_start:test_data_end]
        else:
            test_data = self.data.iloc[test_data_start:test_data_end]

        # Run walk-forward permutation test
        p_value, best_params, real_objective, perm_objectives = walkforward_permutation_test(
            self.strategy, train_data, test_data, objective_func, parameter_grid, n_permutations, show_plot, n_jobs
        )

        # Store test results
        self.walkforward_test_results = {
            'p_value': p_value,
            'best_params': best_params,
            'real_objective': real_objective,
            'objective_func': objective_func.__name__,
            'n_permutations': n_permutations,
            'perm_objectives': perm_objectives
        }

        return self.walkforward_test_results

    def plot_insample_performance(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot in-sample performance.

        Args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        if self.returns is None:
            raise ValueError("No results to plot. Run backtest first.")

        # Calculate cumulative returns
        cum_returns = np.exp(np.cumsum(self.returns)) - 1

        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(cum_returns)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f"In-Sample Performance: {self.strategy.name}")
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Bar Number")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print metrics
        print("\nIn-Sample Performance Metrics:")
        for key, value in self.metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")

    def plot_walkforward_performance(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot walk-forward performance.

        Args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        if not self.walkforward_results:
            raise ValueError("No walk-forward results to plot. Run walk-forward optimization first.")

        # Calculate cumulative returns
        returns = self.walkforward_results['returns']
        cum_returns = np.exp(np.cumsum(returns)) - 1

        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, cum_returns)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f"Walk-Forward Performance: {self.strategy.name}")
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print metrics
        print("\nWalk-Forward Performance Metrics:")
        for key, value in self.walkforward_results['metrics'].items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")

    def save_results(self, filepath: str, include_signals: bool = True):
        """
        Save backtest results to a file.

        Args:
            filepath (str): Path to save the results.
            include_signals (bool, optional): Whether to include signals. Defaults to True.
        """

        # Define a custom JSON encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        # Create results dictionary
        results = {
            'strategy_name': self.strategy.name,
            'strategy_parameters': self.strategy.parameters,
            'metrics': self.metrics,
        }

        # Add optimization results if available
        if self.optimization_results:
            results['optimization_results'] = {
                'best_params': self.optimization_results['best_params'],
                'best_value': float(self.optimization_results['best_value']),
                'objective_func': self.optimization_results['objective_func']
            }

        # Add in-sample test results if available
        if self.insample_test_results:
            results['insample_test_results'] = {
                'p_value': float(self.insample_test_results['p_value']),
                'best_params': self.insample_test_results['best_params'],
                'real_objective': float(self.insample_test_results['real_objective']),
                'objective_func': self.insample_test_results['objective_func'],
                'n_permutations': self.insample_test_results['n_permutations']
            }

        # Add walk-forward results if available
        if self.walkforward_results:
            results['walkforward_results'] = {
                'metrics': self.walkforward_results['metrics'],
                'optimization_params': self.walkforward_results['optimization_params']
            }

        # Add walk-forward test results if available
        if self.walkforward_test_results:
            results['walkforward_test_results'] = {
                'p_value': float(self.walkforward_test_results['p_value']),
                'best_params': self.walkforward_test_results['best_params'],
                'real_objective': float(self.walkforward_test_results['real_objective']),
                'objective_func': self.walkforward_test_results['objective_func'],
                'n_permutations': self.walkforward_test_results['n_permutations']
            }

        # Add signals if requested
        if include_signals and self.signals is not None:
            results['signals'] = self.signals.tolist()

        # Save to file using the custom encoder
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        print(f"Results saved to {filepath}")
