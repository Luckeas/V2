import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union


class PerformancePlotter:
    """
    Utility class for plotting trading strategy performance.
    """
    
    @staticmethod
    def plot_equity_curve(returns: np.ndarray, title: str = "Equity Curve", 
                          figsize: Tuple[int, int] = (12, 6)):
        """
        Plot equity curve from returns.
        
        Args:
            returns (np.ndarray): Array of returns.
            title (str, optional): Plot title. Defaults to "Equity Curve".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        # Calculate cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) - 1
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(cum_returns, label="Strategy")
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(title)
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Bar Number")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_equity_curve_with_time(returns: np.ndarray, dates: Union[pd.DatetimeIndex, List],
                                  title: str = "Equity Curve", figsize: Tuple[int, int] = (12, 6)):
        """
        Plot equity curve from returns with dates.
        
        Args:
            returns (np.ndarray): Array of returns.
            dates (Union[pd.DatetimeIndex, List]): Array or list of dates.
            title (str, optional): Plot title. Defaults to "Equity Curve".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        # Calculate cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) - 1
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(dates, cum_returns, label="Strategy")
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(title)
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_drawdown(returns: np.ndarray, title: str = "Drawdown", 
                      figsize: Tuple[int, int] = (12, 6)):
        """
        Plot drawdown from returns.
        
        Args:
            returns (np.ndarray): Array of returns.
            title (str, optional): Plot title. Defaults to "Drawdown".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        # Calculate cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) - 1
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = 1 - (1 + cum_returns) / (1 + running_max)
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(drawdown, color='red', label="Drawdown")
        plt.title(title)
        plt.ylabel("Drawdown")
        plt.xlabel("Bar Number")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_drawdown_with_time(returns: np.ndarray, dates: Union[pd.DatetimeIndex, List],
                              title: str = "Drawdown", figsize: Tuple[int, int] = (12, 6)):
        """
        Plot drawdown from returns with dates.
        
        Args:
            returns (np.ndarray): Array of returns.
            dates (Union[pd.DatetimeIndex, List]): Array or list of dates.
            title (str, optional): Plot title. Defaults to "Drawdown".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        # Calculate cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) - 1
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = 1 - (1 + cum_returns) / (1 + running_max)
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(dates, drawdown, color='red', label="Drawdown")
        plt.title(title)
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_monthly_returns(returns: np.ndarray, dates: pd.DatetimeIndex, 
                           title: str = "Monthly Returns", figsize: Tuple[int, int] = (12, 6)):
        """
        Plot monthly returns heatmap.
        
        Args:
            returns (np.ndarray): Array of returns.
            dates (pd.DatetimeIndex): Array of dates.
            title (str, optional): Plot title. Defaults to "Monthly Returns".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        # Create DataFrame with returns and dates
        df_returns = pd.DataFrame({'returns': returns}, index=dates)
        
        # Resample to monthly returns
        monthly_returns = df_returns.resample('M').apply(lambda x: np.exp(np.sum(x)) - 1)
        
        # Pivot to create year-month matrix
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        # Create pivot table
        pivot = monthly_returns.pivot_table(index='year', columns='month', values='returns')
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        plt.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
        plt.colorbar(label='Returns')
        plt.title(title)
        plt.ylabel("Year")
        plt.xlabel("Month")
        plt.yticks(pivot.index)
        plt.xticks(pivot.columns, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_signals_on_price(data: pd.DataFrame, signals: np.ndarray, 
                            title: str = "Signals on Price", figsize: Tuple[int, int] = (12, 6)):
        """
        Plot signals on price chart.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray): Array of position signals.
            title (str, optional): Plot title. Defaults to "Signals on Price".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot close price
        plt.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
        
        # Find signal changes
        signal_changes = np.diff(signals, prepend=0)
        
        # Plot long entry points
        long_entries = data.index[signal_changes == 1]
        if len(long_entries) > 0:
            plt.scatter(long_entries, data.loc[long_entries, 'close'], marker='^', color='green', s=100, label='Long Entry')
        
        # Plot long exit points
        long_exits = data.index[(signal_changes == -1) & (np.roll(signals, 1) == 1)]
        if len(long_exits) > 0:
            plt.scatter(long_exits, data.loc[long_exits, 'close'], marker='v', color='red', s=100, label='Long Exit')
        
        # Plot short entry points
        short_entries = data.index[signal_changes == -2]  # From 1 to -1
        if len(short_entries) > 0:
            plt.scatter(short_entries, data.loc[short_entries, 'close'], marker='v', color='red', s=100, label='Short Entry')
        
        # Plot short exit points
        short_exits = data.index[(signal_changes == 2) & (np.roll(signals, 1) == -1)]  # From -1 to 1
        if len(short_exits) > 0:
            plt.scatter(short_exits, data.loc[short_exits, 'close'], marker='^', color='green', s=100, label='Short Exit')
        
        plt.title(title)
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_parameter_sensitivity(strategy, data: pd.DataFrame, param_name: str, 
                                 param_values: List[Any], objective_func: callable,
                                 fixed_params: Dict[str, Any] = None,
                                 title: str = "Parameter Sensitivity", 
                                 figsize: Tuple[int, int] = (12, 6)):
        """
        Plot parameter sensitivity analysis.
        
        Args:
            strategy: Strategy instance.
            data (pd.DataFrame): Market data with OHLC prices.
            param_name (str): Name of the parameter to vary.
            param_values (List[Any]): List of parameter values to test.
            objective_func (callable): Function to evaluate performance.
            fixed_params (Dict[str, Any], optional): Fixed parameters. Defaults to None.
            title (str, optional): Plot title. Defaults to "Parameter Sensitivity".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        """
        objective_values = []
        
        # Loop through parameter values
        for value in param_values:
            # Set parameters
            params = {param_name: value}
            if fixed_params:
                params.update(fixed_params)
            
            # Generate signals
            signals = strategy.generate_signals(data, params)
            
            # Calculate returns
            returns = strategy.compute_returns(data, signals)
            
            # Calculate objective value
            objective_value = objective_func(returns)
            
            objective_values.append(objective_value)
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(param_values, objective_values, marker='o')
        plt.title(f"{title}: {param_name}")
        plt.ylabel(objective_func.__name__)
        plt.xlabel(param_name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
