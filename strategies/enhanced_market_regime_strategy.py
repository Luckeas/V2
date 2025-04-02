import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from itertools import product
from tqdm import tqdm

class Strategy:
    def __init__(self, name: str):
        self.name = name

class EnhancedMarketRegimeStrategy(Strategy):
    def __init__(self,
                 rsi_oversold: int = 35,
                 rsi_overbought: int = 65,
                 volume_multiplier: float = 1.5,
                 max_bars_held: int = 16,
                 bb_window: int = 20,
                 stop_atr_multiplier: float = 1.5):
        super().__init__(name="Enhanced Market Regime Strategy")
        self.parameters = {
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "volume_multiplier": volume_multiplier,
            "max_bars_held": max_bars_held,
            "bb_window": bb_window,
            "stop_atr_multiplier": stop_atr_multiplier
        }

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['middle_band'] = df['close'].rolling(self.parameters['bb_window']).mean()
        df['std_dev'] = df['close'].rolling(self.parameters['bb_window']).std()
        df['upper_band'] = df['middle_band'] + 2 * df['std_dev']
        df['lower_band'] = df['middle_band'] - 2 * df['std_dev']
        df['avg_volume'] = df['volume'].rolling(self.parameters['bb_window']).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['high_low'] = df['high'] - df['low']
        df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        return df

    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        if parameters is not None:
            self.parameters.update(parameters)
        data = self._calculate_indicators(data)
        signals = np.zeros(len(data))
        p = self.parameters
        for i in range(1, len(data)):
            prev_row = data.iloc[i - 1]
            curr_row = data.iloc[i]
            mean_rev_long = (
                prev_row['low'] < prev_row['lower_band'] and
                prev_row['RSI'] < p['rsi_oversold'] and
                prev_row['volume'] > p['volume_multiplier'] * prev_row['avg_volume'] and
                curr_row['close'] > prev_row['low'] * 1.0005
            )
            mean_rev_short = (
                prev_row['high'] > prev_row['upper_band'] and
                prev_row['RSI'] > p['rsi_overbought'] and
                prev_row['volume'] > p['volume_multiplier'] * prev_row['avg_volume'] and
                curr_row['close'] < prev_row['high'] * 0.9995
            )
            if mean_rev_long:
                signals[i] = 1
            elif mean_rev_short:
                signals[i] = -1
        return signals

    def compute_returns(self, data: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        data = self._calculate_indicators(data)
        log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        strategy_returns = np.zeros_like(log_returns)
        position = 0
        entry_bar = 0
        entry_price = 0.0
        for i in range(1, len(data)):
            if position == 0:
                if signals[i] != 0:
                    position = signals[i]
                    entry_bar = i
                    entry_price = data['close'].iloc[i]
                    strategy_returns[i] = position * log_returns.iloc[i]
            else:
                atr = data['atr'].iloc[i - 1]
                stop_loss = entry_price - (position * self.parameters['stop_atr_multiplier'] * atr)
                if (position > 0 and data['close'].iloc[i] < stop_loss) or \
                   (position < 0 and data['close'].iloc[i] > stop_loss):
                    position = 0
                elif i - entry_bar >= self.parameters['max_bars_held']:
                    position = 0
                else:
                    strategy_returns[i] = position * log_returns.iloc[i]
        return strategy_returns

    def optimize(self, data: pd.DataFrame, objective_func: callable,
                 parameter_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], float]:
        best_value = -np.inf if objective_func.__name__ != "drawdown" else np.inf
        best_params = {}
        rsi_oversold = parameter_grid.get('rsi_oversold', [30, 35, 40])
        rsi_overbought = parameter_grid.get('rsi_overbought', [60, 65, 70])
        volume_multiplier = parameter_grid.get('volume_multiplier', [1.2, 1.5, 1.8])
        max_bars_held = parameter_grid.get('max_bars_held', [12, 16, 20])
        bb_window = parameter_grid.get('bb_window', [15, 20, 25])
        stop_atr_multiplier = parameter_grid.get('stop_atr_multiplier', [1.0, 1.5, 2.0])
        total_combinations = (len(rsi_oversold) * len(rsi_overbought) *
                              len(volume_multiplier) * len(max_bars_held) *
                              len(bb_window) * len(stop_atr_multiplier))
        for (rsi_os, rsi_ob, vol_mult, max_bars, bb_win, stop_atr) in tqdm(
            product(rsi_oversold, rsi_overbought, volume_multiplier, max_bars_held, bb_window, stop_atr_multiplier),
            total=total_combinations,
            desc="Optimizing parameters"
        ):
            params = {
                'rsi_oversold': rsi_os,
                'rsi_overbought': rsi_ob,
                'volume_multiplier': vol_mult,
                'max_bars_held': max_bars,
                'bb_window': bb_win,
                'stop_atr_multiplier': stop_atr
            }
            signals = self.generate_signals(data, params)
            returns = self.compute_returns(data, signals)
            value = objective_func(returns)
            if (objective_func.__name__ != "drawdown" and value > best_value) or \
               (objective_func.__name__ == "drawdown" and value < best_value):
                best_value = value
                best_params = params
        return best_params, best_value