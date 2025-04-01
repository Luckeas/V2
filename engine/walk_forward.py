import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import timedelta


class WalkForwardOptimizer:
    """
    Walk-forward optimization as described in the video.
    
    This class implements walk-forward optimization for trading strategies,
    where the strategy is periodically re-optimized on a sliding window of data.
    """
    
    def __init__(
        self, 
        strategy,
        objective_func: callable,
        parameter_grid: Dict[str, List[Any]],
        train_window: Union[int, str] = "4Y",
        train_interval: Union[int, str] = "30D"
    ):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            strategy: Strategy instance to optimize.
            objective_func (callable): Function to maximize during optimization.
            parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.
            train_window (Union[int, str], optional): Training window (bars or time offset). Defaults to "4Y".
            train_interval (Union[int, str], optional): Retraining interval (bars or time offset). Defaults to "30D".
        """
        self.strategy = strategy
        self.objective_func = objective_func
        self.parameter_grid = parameter_grid
        
        if isinstance(train_window, str):
            self.train_window_bars = None
            self.train_window_offset = train_window
        else:
            self.train_window_bars = train_window
            self.train_window_offset = None
        
        if isinstance(train_interval, str):
            self.train_interval_bars = None
            self.train_interval_offset = train_interval
        else:
            self.train_interval_bars = train_interval
            self.train_interval_offset = None
            
        self.optimized_params_history = []
        self.performance_metrics = {}
    
    def run(self, data: pd.DataFrame, show_optimization_progress: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Run walk-forward optimization.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            show_optimization_progress (bool, optional): Whether to show optimization progress. Defaults to True.
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: 
                Signals from walk-forward optimization and DataFrame with optimization results.
        """
        signals = np.zeros(len(data))
        returns = np.zeros(len(data))
        
        # Create result DataFrame
        result_data = []
        
        # Get first optimization point
        if self.train_window_bars:
            first_opt_idx = self.train_window_bars
        else:
            first_opt_idx = data.index.get_indexer(
                [data.index[0] + pd.Timedelta(self.train_window_offset)]
            )[0]
        
        # Set next optimization point
        next_opt_idx = first_opt_idx
        
        # Initialize list to store optimization points
        optimization_points = []
        
        # Progress bar if requested
        iterator = tqdm(range(first_opt_idx, len(data))) if show_optimization_progress else range(first_opt_idx, len(data))
        
        for i in iterator:
            # Check if we need to re-optimize
            if i >= next_opt_idx:
                optimization_points.append(i)
                
                # Get training data
                if self.train_window_bars:
                    train_data = data.iloc[i - self.train_window_bars:i]
                else:
                    train_start = data.index[i] - pd.Timedelta(self.train_window_offset)
                    train_start_idx = data.index.get_indexer([train_start])[0]
                    train_data = data.iloc[train_start_idx:i]
                
                # Optimize strategy on training data
                best_params, best_value = self.strategy.optimize(
                    train_data, self.objective_func, self.parameter_grid
                )
                
                # Store optimization results
                self.optimized_params_history.append({
                    'index': i,
                    'date': data.index[i],
                    'params': best_params,
                    'train_objective': best_value
                })
                
                # Add to result data
                result_data.append({
                    'date': data.index[i],
                    'train_objective': best_value,
                    **best_params
                })
                
                # Set next optimization point
                if self.train_interval_bars:
                    next_opt_idx = i + self.train_interval_bars
                else:
                    next_date = data.index[i] + pd.Timedelta(self.train_interval_offset)
                    next_idx = data.index.get_indexer([next_date])[0]
                    next_opt_idx = next_idx if next_idx >= 0 else len(data)
            
            # Generate signal for current bar using latest optimized parameters
            if self.optimized_params_history:
                latest_params = self.optimized_params_history[-1]['params']
                current_bar_data = data.iloc[:i+1]
                signals[i] = self.strategy.generate_signals(current_bar_data, latest_params)[-1]
                
                # Calculate return for the bar
                if i > 0:
                    log_return = np.log(data.iloc[i]['close'] / data.iloc[i-1]['close'])
                    returns[i] = signals[i-1] * log_return
        
        # Convert result data to DataFrame
        results_df = pd.DataFrame(result_data)
        if not results_df.empty:
            results_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        self.performance_metrics = {
            'objective_value': self.objective_func(returns[first_opt_idx:]),
            'cumulative_return': np.exp(np.sum(returns[first_opt_idx:])) - 1,
            'annualized_return': np.exp(np.sum(returns[first_opt_idx:]) * 252 / len(returns[first_opt_idx:])) - 1,
            'annualized_volatility': np.std(returns[first_opt_idx:]) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns[first_opt_idx:]) / np.std(returns[first_opt_idx:]) * np.sqrt(252) if np.std(returns[first_opt_idx:]) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns[first_opt_idx:]),
            'optimization_points': optimization_points
        }
        
        return signals, results_df
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate the maximum drawdown from returns.
        
        Args:
            returns (np.ndarray): Array of returns.
            
        Returns:
            float: Maximum drawdown.
        """
        # Calculate cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) - 1
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = 1 - (1 + cum_returns) / (1 + running_max)
        
        # Return maximum drawdown
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    def plot_optimization_results(self, data: pd.DataFrame, signals: np.ndarray = None):
        """
        Plot optimization results.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray, optional): Signals from walk-forward optimization. Defaults to None.
        """
        if not self.optimized_params_history:
            print("No optimization results to plot.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot close price
        axes[0].plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
        axes[0].set_ylabel('Price')
        axes[0].set_title('Walk-Forward Optimization Results')
        
        # Highlight optimization points
        for opt_point in self.optimized_params_history:
            axes[0].axvline(x=data.index[opt_point['index']], color='red', linestyle='--', alpha=0.3)
        
        # Plot cumulative returns if signals are provided
        if signals is not None:
            # Calculate returns
            returns = np.zeros(len(data))
            for i in range(1, len(data)):
                log_return = np.log(data.iloc[i]['close'] / data.iloc[i-1]['close'])
                returns[i] = signals[i-1] * log_return
            
            # Calculate cumulative returns
            cum_returns = np.exp(np.cumsum(returns)) - 1
            
            # Plot cumulative returns
            axes[1].plot(data.index, cum_returns, label='Cumulative Returns', color='green')
            axes[1].set_ylabel('Cumulative Returns')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        for key, value in self.performance_metrics.items():
            if key != 'optimization_points':
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        # Plot parameter evolution if there are parameters to plot
        if self.optimized_params_history:
            param_keys = list(self.optimized_params_history[0]['params'].keys())
            if param_keys:
                fig, axes = plt.subplots(len(param_keys), 1, figsize=(12, 3 * len(param_keys)))
                if len(param_keys) == 1:
                    axes = [axes]  # Make it iterable when there's only one parameter
                
                for i, param in enumerate(param_keys):
                    param_values = [opt['params'][param] for opt in self.optimized_params_history]
                    dates = [opt['date'] for opt in self.optimized_params_history]
                    axes[i].plot(dates, param_values, marker='o')
                    axes[i].set_ylabel(param)
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
