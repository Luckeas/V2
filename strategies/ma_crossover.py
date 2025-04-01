import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from itertools import product

from strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Goes long when the fast MA crosses above the slow MA.
    Goes short when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            fast_period (int, optional): Period for fast MA. Defaults to 10.
            slow_period (int, optional): Period for slow MA. Defaults to 50.
        """
        super().__init__(name="Moving Average Crossover")
        self.parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
    
    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.
            
        Returns:
            np.ndarray: Array of position signals (1 for long, 0 for flat, -1 for short).
        """
        if parameters is not None:
            self.parameters.update(parameters)
        
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        # Calculate moving averages
        data["fast_ma"] = data["close"].rolling(window=fast_period).mean()
        data["slow_ma"] = data["close"].rolling(window=slow_period).mean()
        
        # Initialize signals array
        signals = np.zeros(len(data))
        
        # Generate signals based on MA crossover
        # 1 when fast_ma > slow_ma, -1 when fast_ma < slow_ma
        signals[fast_period:] = np.where(
            data["fast_ma"][fast_period:] > data["slow_ma"][fast_period:], 1, -1
        )
        
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
        
        fast_periods = parameter_grid.get("fast_period", [self.parameters["fast_period"]])
        slow_periods = parameter_grid.get("slow_period", [self.parameters["slow_period"]])
        
        for fast_period, slow_period in product(fast_periods, slow_periods):
            # Skip invalid combinations
            if fast_period >= slow_period:
                continue
                
            params = {
                "fast_period": fast_period,
                "slow_period": slow_period
            }
            
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
