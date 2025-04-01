from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All custom strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the strategy.
        
        Args:
            name (str): Name of the strategy.
        """
        self.name = name
        self.parameters = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate trading signals based on the provided data and parameters.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.
            
        Returns:
            np.ndarray: Array of position signals (1 for long, 0 for flat, -1 for short).
        """
        pass
    
    @abstractmethod
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
        pass
    
    def compute_returns(self, data: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        """
        Compute strategy returns based on the signals and price data.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray): Array of position signals.
            
        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Compute log returns (close-to-close)
        log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        
        # Shift signals forward by 1 bar to avoid look-ahead bias
        shifted_signals = np.roll(signals, 1)
        shifted_signals[0] = 0  # No position for the first bar
        
        # Compute strategy returns
        strategy_returns = shifted_signals * log_returns
        
        return strategy_returns
