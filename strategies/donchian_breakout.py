import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from itertools import product

from strategies.base import Strategy


class DonchianBreakout(Strategy):
    """
    Donchian Channel Breakout strategy.
    
    Goes long when the current close is the highest over a given lookback period.
    Goes short when the current close is the lowest over a given lookback period.
    """
    
    def __init__(self, lookback: int = 20):
        """
        Initialize the Donchian Breakout strategy.
        
        Args:
            lookback (int, optional): Lookback period for calculating channel. Defaults to 20.
        """
        super().__init__(name="Donchian Breakout")
        self.parameters = {"lookback": lookback}
    
    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate trading signals based on Donchian channel breakouts.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.
            
        Returns:
            np.ndarray: Array of position signals (1 for long, 0 for flat, -1 for short).
        """
        if parameters is not None:
            self.parameters.update(parameters)
        
        lookback = self.parameters["lookback"]
        close = data["close"].values
        signals = np.zeros(len(close))
        
        # Calculate rolling highest high and lowest low
        for i in range(lookback, len(close)):
            window = close[i-lookback:i]
            if close[i] > np.max(window):
                signals[i] = 1  # Long signal
            elif close[i] < np.min(window):
                signals[i] = -1  # Short signal
        
        return signals
    
    def optimize(self, data: pd.DataFrame, objective_func: callable, 
                parameter_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], float]:
        """
        Optimize strategy parameters based on the provided data and objective function.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            objective_func (callable): Function to maximize/minimize during optimization.
            parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.
            
        Returns:
            Tuple[Dict[str, Any], float]: Best parameters and the corresponding objective value.
        """
        best_value = -np.inf if objective_func.__name__ != "drawdown" else np.inf
        best_params = {}
        
        lookback_values = parameter_grid.get("lookback", [self.parameters["lookback"]])
        
        for lookback in lookback_values:
            params = {"lookback": lookback}
            signals = self.generate_signals(data, params)
            returns = self.compute_returns(data, signals)
            
            # Calculate objective function value
            value = objective_func(returns)
            
            # Update best parameters if needed
            if (objective_func.__name__ != "drawdown" and value > best_value) or \
               (objective_func.__name__ == "drawdown" and value < best_value):
                best_value = value
                best_params = params.copy()
        
        return best_params, best_value
