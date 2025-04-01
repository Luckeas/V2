import numpy as np
import pandas as pd
from typing import Union


def profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the profit factor.
    
    Profit factor = sum(positive returns) / abs(sum(negative returns))
    
    Args:
        returns (Union[np.ndarray, pd.Series]): Array or series of returns.
        
    Returns:
        float: Profit factor.
    """
    positive_returns = returns[returns > 0].sum()
    negative_returns = abs(returns[returns < 0].sum())
    
    # Handle case where there are no negative returns
    if negative_returns == 0:
        return float('inf') if positive_returns > 0 else 0
    
    return positive_returns / negative_returns


def sharpe_ratio(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Sharpe ratio = (mean(returns) - risk_free_rate) / std(returns) * sqrt(periods_per_year)
    
    Args:
        returns (Union[np.ndarray, pd.Series]): Array or series of returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252.
        
    Returns:
        float: Annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Handle case where std is 0
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return * np.sqrt(periods_per_year)


def drawdown(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the maximum drawdown.
    
    Maximum drawdown = max(1 - current_value / peak_value)
    
    Args:
        returns (Union[np.ndarray, pd.Series]): Array or series of returns.
        
    Returns:
        float: Maximum drawdown as a positive value.
    """
    # Calculate cumulative returns
    cum_returns = np.exp(np.cumsum(returns)) - 1
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = 1 - (1 + cum_returns) / (1 + running_max)
    
    # Return maximum drawdown
    return np.max(drawdown)


def cagr(returns: Union[np.ndarray, pd.Series], periods_per_year: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    
    CAGR = (final_value / initial_value)^(1 / years) - 1
    
    Args:
        returns (Union[np.ndarray, pd.Series]): Array or series of returns.
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252.
        
    Returns:
        float: CAGR.
    """
    # Calculate cumulative returns
    cum_return = np.exp(np.sum(returns)) - 1
    
    # Calculate number of years
    years = len(returns) / periods_per_year
    
    # Calculate CAGR
    return (1 + cum_return) ** (1 / years) - 1


def calmar_ratio(returns: Union[np.ndarray, pd.Series], periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio.
    
    Calmar ratio = CAGR / maximum drawdown
    
    Args:
        returns (Union[np.ndarray, pd.Series]): Array or series of returns.
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252.
        
    Returns:
        float: Calmar ratio.
    """
    max_dd = drawdown(returns)
    
    # Handle case where drawdown is 0
    if max_dd == 0:
        return float('inf')
    
    return cagr(returns, periods_per_year) / max_dd


def sortino_ratio(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino ratio.
    
    Sortino ratio = (mean(returns) - risk_free_rate) / std(negative returns) * sqrt(periods_per_year)
    
    Args:
        returns (Union[np.ndarray, pd.Series]): Array or series of returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252.
        
    Returns:
        float: Sortino ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Calculate downside deviation (standard deviation of negative returns)
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    downside_deviation = np.std(negative_returns)
    
    return (mean_return - risk_free_rate) / downside_deviation * np.sqrt(periods_per_year)
