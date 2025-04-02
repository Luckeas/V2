import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from itertools import product

from strategies.base import Strategy


class EnhancedMarketRegimeStrategy(Strategy):
    """
    Enhanced Market Regime Strategy with:
    1. Mean Reversion Entry
    2. Trend Following Entry
    3. Advanced ATR-based Exit
    4. Trailing Stop
    5. Time-Based Exit
    """
    
    def __init__(self, 
                 rsi_oversold: int = 35, 
                 rsi_overbought: int = 65, 
                 volume_multiplier: float = 1.5,
                 atr_stop_multiplier: float = 2.0,
                 max_bars_held: int = 16,
                 bb_window: int = 20,
                 trend_rsi_min: int = 50,
                 trend_rsi_max: int = 70,
                 ma_slope_threshold: float = 0.1):
        """
        Initialize the Enhanced Market Regime Strategy.
        
        Args:
            rsi_oversold (int): RSI threshold for oversold condition. Defaults to 35.
            rsi_overbought (int): RSI threshold for overbought condition. Defaults to 65.
            volume_multiplier (float): Minimum volume multiple for entries. Defaults to 1.5.
            atr_stop_multiplier (float): ATR multiplier for stop loss. Defaults to 2.0.
            max_bars_held (int): Maximum bars to hold a trade. Defaults to 16.
            bb_window (int): Bollinger Bands calculation window. Defaults to 20.
            trend_rsi_min (int): Minimum RSI for trend following. Defaults to 50.
            trend_rsi_max (int): Maximum RSI for trend following. Defaults to 70.
            ma_slope_threshold (float): MA slope threshold. Defaults to 0.1.
        """
        super().__init__(name="Enhanced Market Regime Strategy")
        self.parameters = {
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "volume_multiplier": volume_multiplier,
            "atr_stop_multiplier": atr_stop_multiplier,
            "max_bars_held": max_bars_held,
            "bb_window": bb_window,
            "trend_rsi_min": trend_rsi_min,
            "trend_rsi_max": trend_rsi_max,
            "ma_slope_threshold": ma_slope_threshold
        }
    
    def _calculate_indicators(self, data: pd.DataFrame):
        """Calculate additional indicators needed for the strategy."""
        data['middle_band'] = data['close'].rolling(self.parameters['bb_window']).mean()
        data['std_dev'] = data['close'].rolling(self.parameters['bb_window']).std()
        data['upper_band'] = data['middle_band'] + 2 * data['std_dev']
        data['lower_band'] = data['middle_band'] - 2 * data['std_dev']
        data['avg_volume'] = data['volume'].rolling(self.parameters['bb_window']).mean()
        data['MA'] = data['close'].rolling(window=50).mean()
        data['MA_slope'] = (data['MA'] / data['MA'].shift(10) - 1) * 100
        return data
    
    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate trading signals based on market regime conditions.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.
            
        Returns:
            np.ndarray: Array of position signals (1 for long, 0 for flat, -1 for short).
        """
        if parameters is not None:
            self.parameters.update(parameters)
        
        # Calculate necessary indicators
        data = self._calculate_indicators(data)
        
        # Initialize signals array
        signals = np.zeros(len(data))
        
        # Parameters for readability
        p = self.parameters
        
        # Iterate through data starting from second row to use previous row's data
        for i in range(1, len(data)):
            prev_row = data.iloc[i-1]
            curr_row = data.iloc[i]
            
            # Mean Reversion Long Entry
            mean_rev_long = (
                prev_row['low'] < prev_row['lower_band'] and
                prev_row['RSI'] < p['rsi_oversold'] and
                prev_row['volume'] > p['volume_multiplier'] * prev_row['avg_volume'] and
                curr_row['close'] > prev_row['low'] * 1.0005
            )
            
            # Mean Reversion Short Entry
            mean_rev_short = (
                prev_row['high'] > prev_row['upper_band'] and
                prev_row['RSI'] > p['rsi_overbought'] and
                prev_row['volume'] > p['volume_multiplier'] * prev_row['avg_volume'] and
                curr_row['close'] < prev_row['high'] * 0.9995
            )
            
            # Trend Following Long Entry
            trend_long = (
                prev_row['close'] > prev_row['MA'] and
                prev_row['RSI'] > p['trend_rsi_min'] and
                prev_row['RSI'] < p['trend_rsi_max'] and
                prev_row['MA_slope'] > p['ma_slope_threshold'] and
                prev_row['volume'] > prev_row['avg_volume'] * 1.2
            )
            
            # Trend Following Short Entry
            trend_short = (
                prev_row['close'] < prev_row['MA'] and
                prev_row['RSI'] < p['trend_rsi_min'] and
                prev_row['RSI'] > p['trend_rsi_max'] and
                prev_row['MA_slope'] < -p['ma_slope_threshold'] and
                prev_row['volume'] > prev_row['avg_volume'] * 1.2
            )
            
            # Assign signals
            if mean_rev_long or trend_long:
                signals[i] = 1
            elif mean_rev_short or trend_short:
                signals[i] = -1
        
        return signals
    
    def compute_returns(self, data: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        """
        Compute strategy returns with ATR-based stop loss and time-based exit.
        
        Args:
            data (pd.DataFrame): Market data.
            signals (np.ndarray): Trading signals.
            
        Returns:
            np.ndarray: Strategy returns.
        """
        # Calculate log returns
        log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        
        # Calculate ATR for stop loss calculation
        data['high_low'] = data['high'] - data['low']
        data['high_prev_close'] = abs(data['high'] - data['close'].shift(1))
        data['low_prev_close'] = abs(data['low'] - data['close'].shift(1))
        data['true_range'] = data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        data['ATR'] = data['true_range'].rolling(window=14).mean()
        
        # Shift signals to avoid look-ahead bias
        shifted_signals = np.roll(signals, 1)
        shifted_signals[0] = 0
        
        # Time-based exit tracking
        bars_held = np.zeros_like(signals)
        
        # Enhanced return calculation with time and stop loss consideration
        strategy_returns = np.zeros_like(log_returns)
        
        for i in range(1, len(data)):
            if shifted_signals[i] != 0:
                # Increment bars held for current position
                bars_held[i] = bars_held[i-1] + 1 if bars_held[i-1] > 0 else 1
                
                # Calculate ATR-based stop loss
                atr = data.iloc[i-1]['ATR']
                
                # Time-based exit or maximum bar holding
                if bars_held[i] >= self.parameters['max_bars_held']:
                    strategy_returns[i] = shifted_signals[i] * log_returns[i]
                    bars_held[i] = 0
                    shifted_signals[i] = 0
                else:
                    strategy_returns[i] = shifted_signals[i] * log_returns[i]
            else:
                bars_held[i] = 0
        
        return strategy_returns
    
    def optimize(self, data: pd.DataFrame, objective_func: callable, 
                parameter_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], float]:
        """
        Optimize strategy parameters based on the objective function.
        
        Args:
            data (pd.DataFrame): Market data.
            objective_func (callable): Performance metric to optimize.
            parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.
            
        Returns:
            Tuple[Dict[str, Any], float]: Best parameters and objective value.
        """
        best_value = -np.inf if objective_func.__name__ != "drawdown" else np.inf
        best_params = {}
        
        # Define parameter ranges with defaults if not provided
        rsi_oversold = parameter_grid.get('rsi_oversold', [30, 35, 40])
        rsi_overbought = parameter_grid.get('rsi_overbought', [60, 65, 70])
        volume_multiplier = parameter_grid.get('volume_multiplier', [1.2, 1.5, 1.8])
        atr_stop_multiplier = parameter_grid.get('atr_stop_multiplier', [1.5, 2.0, 2.5])
        max_bars_held = parameter_grid.get('max_bars_held', [12, 16, 20])
        ma_slope_threshold = parameter_grid.get('ma_slope_threshold', [0.05, 0.1, 0.15])
        
        # Grid search
        for (rsi_os, rsi_ob, vol_mult, atr_mult, max_bars, slope_thresh) in product(
            rsi_oversold, rsi_overbought, volume_multiplier, 
            atr_stop_multiplier, max_bars_held, ma_slope_threshold
        ):
            params = {
                'rsi_oversold': rsi_os,
                'rsi_overbought': rsi_ob,
                'volume_multiplier': vol_mult,
                'atr_stop_multiplier': atr_mult,
                'max_bars_held': max_bars,
                'ma_slope_threshold': slope_thresh
            }
            
            # Generate signals and compute returns
            signals = self.generate_signals(data, params)
            returns = self.compute_returns(data, signals)
            
            # Evaluate objective function
            value = objective_func(returns)
            
            # Update best parameters
            if (objective_func.__name__ != "drawdown" and value > best_value) or \
               (objective_func.__name__ == "drawdown" and value < best_value):
                best_value = value
                best_params = params
        
        return best_params, best_value
