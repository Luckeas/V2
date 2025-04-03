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

    def _calculate_support_resistance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based support and resistance levels."""
        df = data.copy()
        lookback_period = min(100, len(df))

        # For each bar, we'll identify potential support/resistance levels
        # but only using data available before that bar
        for i in range(lookback_period, len(df)):
            # Use the last 100 bars before the current bar
            historical_data = df.iloc[i - lookback_period:i]

            # Create price buckets (divide price range into 20 buckets)
            price_min = historical_data['low'].min()
            price_max = historical_data['high'].max()
            price_range = price_max - price_min

            if price_range == 0:  # Avoid division by zero
                df.loc[df.index[i], 'closest_support'] = np.nan
                df.loc[df.index[i], 'closest_resistance'] = np.nan
                continue

            bucket_size = price_range / 20

            # Initialize volume profile
            volume_profile = {}

            # Create price buckets and sum volume for each
            for _, row in historical_data.iterrows():
                # Skip rows with missing volume
                if pd.isna(row['volume']) or row['volume'] == 0:
                    continue

                # Price range that this bar spans
                for price in np.arange(row['low'], row['high'] + bucket_size, bucket_size):
                    bucket = round(price / bucket_size) * bucket_size
                    if bucket in volume_profile:
                        volume_profile[bucket] += row['volume']
                    else:
                        volume_profile[bucket] = row['volume']

            if not volume_profile:  # No valid prices or volume
                df.loc[df.index[i], 'closest_support'] = np.nan
                df.loc[df.index[i], 'closest_resistance'] = np.nan
                continue

            # Find high volume nodes (potential support/resistance)
            sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)

            # Take top 3 volume nodes as support/resistance (or fewer if not enough)
            top_count = min(3, len(sorted_profile))
            support_resistance_levels = [price for price, _ in sorted_profile[:top_count]]

            # Tag with current price
            current_price = df['close'].iloc[i]

            # Find closest level above and below current price
            levels_above = [l for l in support_resistance_levels if l > current_price]
            levels_below = [l for l in support_resistance_levels if l < current_price]

            closest_resistance = min(levels_above) if levels_above else np.nan
            closest_support = max(levels_below) if levels_below else np.nan

            # Add to dataframe
            df.loc[df.index[i], 'closest_support'] = closest_support
            df.loc[df.index[i], 'closest_resistance'] = closest_resistance

        return df

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Original indicators
        df['middle_band'] = df['close'].rolling(self.parameters['bb_window']).mean()
        df['std_dev'] = df['close'].rolling(self.parameters['bb_window']).std()
        df['upper_band'] = df['middle_band'] + 2 * df['std_dev']
        df['lower_band'] = df['middle_band'] - 2 * df['std_dev']
        df['avg_volume'] = df['volume'].rolling(self.parameters['bb_window']).mean()

        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR calculation
        df['high_low'] = df['high'] - df['low']
        df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()

        # NEW: Add trend indicators
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()

        # NEW: Calculate ADX for trend strength
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DI and -DI
        plus_dm = high.diff()
        minus_dm = low.diff().multiply(-1)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True ranges
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ADX
        smoothing_period = 14
        tr14 = tr.rolling(window=smoothing_period).sum()

        # Handle division by zero
        tr14_safe = tr14.replace(0, np.nan)

        plus_di14 = 100 * (plus_dm.rolling(window=smoothing_period).sum() / tr14_safe)
        minus_di14 = 100 * (minus_dm.rolling(window=smoothing_period).sum() / tr14_safe)

        # Handle division by zero in DX calculation
        di_sum = plus_di14 + minus_di14
        di_sum_safe = di_sum.replace(0, np.nan)

        dx = 100 * abs(plus_di14 - minus_di14) / di_sum_safe
        df['adx'] = dx.rolling(window=smoothing_period).mean()

        # NEW: Add volatility indicators
        # ATR Percentile (is current ATR high compared to recent history?)
        atr_lookback = 100
        df['atr_percentile'] = df['atr'].rolling(window=atr_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        ).fillna(0.5)  # Default to 0.5 for missing values

        # Bollinger Band Width
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band'].replace(0, np.nan)

        # Historical Volatility
        returns = np.log(df['close'] / df['close'].shift(1))
        df['historical_vol'] = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized

        return df

    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        if parameters is not None:
            self.parameters.update(parameters)

        # Add default parameters for new filters if not present
        if 'adx_threshold' not in self.parameters:
            self.parameters['adx_threshold'] = 25  # ADX above this indicates strong trend
        if 'max_volatility_percentile' not in self.parameters:
            self.parameters['max_volatility_percentile'] = 0.8  # Avoid top 20% volatile periods
        if 'respect_sma' not in self.parameters:
            self.parameters['respect_sma'] = True  # Respect SMA trend direction

        # Calculate all indicators
        data = self._calculate_indicators(data)

        # Calculate support/resistance levels
        data = self._calculate_support_resistance(data)

        signals = np.zeros(len(data))
        p = self.parameters

        # Use only data available at bar open
        for i in range(2, len(data)):  # Start at index 2 to have enough history
            prev_row = data.iloc[i - 1]
            earlier_row = data.iloc[i - 2]

            # Skip if missing data
            if pd.isna(prev_row['adx']) or pd.isna(prev_row['atr_percentile']):
                continue

            # Check if volatility is too high
            volatility_too_high = prev_row['atr_percentile'] > p['max_volatility_percentile']

            # Check trend direction
            trend_down = prev_row['sma50'] < prev_row['sma200'] if not pd.isna(prev_row['sma50']) and not pd.isna(
                prev_row['sma200']) else False
            strong_trend = prev_row['adx'] > p['adx_threshold'] if not pd.isna(prev_row['adx']) else False

            # Basic mean reversion signals (from existing code)
            mean_rev_long_base = (
                    earlier_row['low'] < earlier_row['lower_band'] and
                    earlier_row['RSI'] < p['rsi_oversold'] and
                    earlier_row['volume'] > p['volume_multiplier'] * earlier_row['avg_volume'] and
                    prev_row['open'] > earlier_row['low'] * 1.0005
            )

            mean_rev_short_base = (
                    earlier_row['high'] > earlier_row['upper_band'] and
                    earlier_row['RSI'] > p['rsi_overbought'] and
                    earlier_row['volume'] > p['volume_multiplier'] * earlier_row['avg_volume'] and
                    prev_row['open'] < earlier_row['high'] * 0.9995
            )

            # Apply filters for long trades
            mean_rev_long = mean_rev_long_base and not volatility_too_high

            # For long trades, either don't need to respect trend or trend should be up
            if p['respect_sma']:
                mean_rev_long = mean_rev_long and (not strong_trend or not trend_down)

            # Check if price is near support level (if available)
            if not pd.isna(prev_row['closest_support']):
                support_proximity = abs(prev_row['low'] - prev_row['closest_support']) / prev_row['closest_support']
                mean_rev_long = mean_rev_long and support_proximity < 0.01  # Within 1% of support

            # Apply filters for short trades
            mean_rev_short = mean_rev_short_base and not volatility_too_high

            # For short trades, either don't need to respect trend or trend should be down
            if p['respect_sma']:
                mean_rev_short = mean_rev_short and (not strong_trend or trend_down)

            # Check if price is near resistance level (if available)
            if not pd.isna(prev_row['closest_resistance']):
                resistance_proximity = abs(prev_row['high'] - prev_row['closest_resistance']) / prev_row[
                    'closest_resistance']
                mean_rev_short = mean_rev_short and resistance_proximity < 0.01  # Within 1% of resistance

            # Set signals based on conditions
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

        for i in range(1, len(data) - 1):  # Changed to len(data)-1 to prevent index out of range
            # Apply current position to next bar's returns
            if position != 0:
                strategy_returns[i + 1] = position * log_returns.iloc[i + 1]

            # Check for position exit
            if position != 0:
                atr = data['atr'].iloc[i - 1]
                stop_loss = entry_price - (position * self.parameters['stop_atr_multiplier'] * atr)
                if (position > 0 and data['close'].iloc[i] < stop_loss) or \
                        (position < 0 and data['close'].iloc[i] > stop_loss):
                    position = 0
                elif i - entry_bar >= self.parameters['max_bars_held']:
                    position = 0

            # Check for new entry if not in position
            if position == 0 and signals[i] != 0:
                position = signals[i]
                entry_bar = i
                entry_price = data['close'].iloc[i]
                # Note: Returns for this new position will be applied on the next iteration

        return strategy_returns

    def optimize(self, data: pd.DataFrame, objective_func: callable,
                 parameter_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], float]:
        from joblib import Parallel, delayed
        import itertools

        best_value = -np.inf if objective_func.__name__ != "drawdown" else np.inf
        best_params = {}

        # Extract parameter values from grid
        rsi_oversold = parameter_grid.get('rsi_oversold', [30, 35, 40])
        rsi_overbought = parameter_grid.get('rsi_overbought', [60, 65, 70])
        volume_multiplier = parameter_grid.get('volume_multiplier', [1.2, 1.5, 1.8])
        max_bars_held = parameter_grid.get('max_bars_held', [2, 3, 5, 10])
        bb_window = parameter_grid.get('bb_window', [15, 20, 25])
        stop_atr_multiplier = parameter_grid.get('stop_atr_multiplier', [1.0, 1.5, 2.0])

        # Create all parameter combinations
        param_combinations = list(itertools.product(
            rsi_oversold, rsi_overbought, volume_multiplier,
            max_bars_held, bb_window, stop_atr_multiplier
        ))

        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations in parallel...")

        # Define function to evaluate a single parameter set
        def evaluate_params(params_tuple):
            rsi_os, rsi_ob, vol_mult, max_bars, bb_win, stop_atr = params_tuple

            # Skip invalid combinations
            if rsi_os >= rsi_ob:
                return None, -np.inf if objective_func.__name__ != "drawdown" else np.inf

            params = {
                'rsi_oversold': rsi_os,
                'rsi_overbought': rsi_ob,
                'volume_multiplier': vol_mult,
                'max_bars_held': max_bars,
                'bb_window': bb_win,
                'stop_atr_multiplier': stop_atr
            }

            try:
                signals = self.generate_signals(data, params)
                returns = self.compute_returns(data, signals)
                value = objective_func(returns)
                return params, value
            except Exception as e:
                print(f"Error with params {params}: {e}")
                return None, -np.inf if objective_func.__name__ != "drawdown" else np.inf

        # Run evaluations in parallel with 8 jobs
        results = Parallel(n_jobs=8, verbose=10)(
            delayed(evaluate_params)(params) for params in param_combinations
        )

        # Find best results
        for params, value in results:
            if params is not None:
                if (objective_func.__name__ != "drawdown" and value > best_value) or \
                        (objective_func.__name__ == "drawdown" and value < best_value):
                    best_value = value
                    best_params = params

        return best_params, best_value