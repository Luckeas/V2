import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional
from itertools import product
import sys
import time
import numpy as np

from strategies.base import Strategy
from utils.candlestick_patterns import CandlestickPatternDetector


class MorningReversal(Strategy):
    """
    Morning Reversal strategy.

    Trading strategy that looks for mean reversion opportunities during the volatile morning session.
    It uses Bollinger Bands, VWAP, candlestick patterns and key levels to make trading decisions.

    Entry Conditions:
    - Time window: Between 9:30 AM and 10:00 AM Eastern Time
    - Long: Price below lower Bollinger Band and VWAP with bullish reversal pattern
    - Short: Price above upper Bollinger Band and VWAP with bearish reversal pattern

    Exit Conditions:
    - Long: Exit at the nearest resistance level (previous day high or post-hours high)
    - Short: Exit at the nearest support level (previous day low or post-hours low)

    Stop Loss:
    - Long: Entry Price - 1.5 * ATR(14)
    - Short: Entry Price + 1.5 * ATR(14)
    """

    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 atr_period: int = 14,
                 stop_atr_multiplier: float = 1.5,
                 risk_per_trade: float = 0.01,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 atr_min_threshold: float = 0.8,
                 atr_max_threshold: float = 2.0,
                 bullish_patterns: List[str] = None,
                 bearish_patterns: List[str] = None):
        """
        Initialize the Morning Reversal strategy.

        Args:
            bb_period (int, optional): Period for Bollinger Bands. Defaults to 20.
            bb_std (float, optional): Standard deviation for Bollinger Bands. Defaults to 2.0.
            atr_period (int, optional): Period for ATR calculation. Defaults to 14.
            stop_atr_multiplier (float, optional): Multiplier for ATR-based stop loss. Defaults to 1.5.
            risk_per_trade (float, optional): Risk per trade as fraction of account. Defaults to 0.01 (1%).
            rsi_period (int, optional): Period for RSI calculation. Defaults to 14.
            rsi_oversold (int, optional): RSI threshold for oversold condition. Defaults to 30.
            rsi_overbought (int, optional): RSI threshold for overbought condition. Defaults to 70.
            atr_min_threshold (float, optional): Minimum ATR ratio threshold for entry. Defaults to 0.8.
            atr_max_threshold (float, optional): Maximum ATR ratio threshold for entry. Defaults to 2.0.
            bullish_patterns (List[str], optional): List of bullish candlestick patterns to use.
                                                   Defaults to None (uses a preset list).
            bearish_patterns (List[str], optional): List of bearish candlestick patterns to use.
                                                   Defaults to None (uses a preset list).
        """
        super().__init__(name="Morning Reversal")

        # Strategy parameters
        self.parameters = {
            "bb_period": bb_period,
            "bb_std": bb_std,
            "atr_period": atr_period,
            "stop_atr_multiplier": stop_atr_multiplier,
            "risk_per_trade": risk_per_trade,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "atr_min_threshold": atr_min_threshold,
            "atr_max_threshold": atr_max_threshold
        }

        # Default bullish patterns
        if bullish_patterns is None:
            self.bullish_patterns = [
                'hammer', 'engulfing', 'morning_star', 'piercing_line',
                'three_white_soldiers', 'three_outside_up', 'morning_doji_star'
            ]
        else:
            self.bullish_patterns = bullish_patterns

        # Default bearish patterns
        if bearish_patterns is None:
            self.bearish_patterns = [
                'shooting_star', 'engulfing', 'evening_star', 'dark_cloud_cover',
                'three_black_crows', 'three_line_strike_bullish'
            ]
        else:
            self.bearish_patterns = bearish_patterns

        # Cache for indicators and patterns to avoid recalculation
        self._indicator_cache = {}
        self._pattern_cache = {}
        self._key_levels_cache = None
        self._levels_already_logged = False  # Flag to prevent repeated logging

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            pd.DataFrame: Data with added technical indicators.
        """
        # Check if indicators already calculated
        data_key = id(data)
        if data_key in self._indicator_cache:
            return self._indicator_cache[data_key]

        df = data.copy()

        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.parameters['bb_period']).mean()
        df['std'] = df['close'].rolling(window=self.parameters['bb_period']).std()
        df['bb_upper'] = df['sma'] + (df['std'] * self.parameters['bb_std'])
        df['bb_lower'] = df['sma'] - (df['std'] * self.parameters['bb_std'])

        # Calculate VWAP if volume exists, otherwise use SMA as substitute
        if 'volume' in df.columns:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        else:
            # If no volume data, use SMA as a substitute
            df['vwap'] = df['close'].rolling(window=self.parameters['bb_period']).mean()

        # Calculate ATR for stop loss and entry filter
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        df['atr'] = tr.rolling(window=self.parameters['atr_period']).mean()

        # Calculate ATR ratio (current ATR / average ATR)
        # This helps identify periods of increased volatility
        df['atr_ratio'] = tr / df['atr']

        # Calculate RSI for entry filter
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Extract time component if datetime index exists
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute

            # Create a flag for morning session (9:30 AM to 10:00 AM)
            morning_session = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] == 10) & (df['minute'] == 0))
            df['morning_session'] = morning_session
        else:
            # If no datetime index, assume all bars are valid
            df['morning_session'] = True

        # Store in cache
        self._indicator_cache[data_key] = df

        return df

    def _get_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key price levels for support, resistance, and targets.

        For MES futures data, this identifies:
        1. Previous day's high and low from regular trading hours (RTH)
        2. Post-hours high and low from extended trading hours (ETH)

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            Dict[str, float]: Dictionary with key levels.
        """
        # Use cached levels if available
        if self._key_levels_cache is not None:
            return self._key_levels_cache

        levels = {}

        # Determine if we should log levels (avoid excessive logging)
        should_log = (not hasattr(self, '_levels_logged') or not self._levels_logged)

        if isinstance(data.index, pd.DatetimeIndex):
            try:
                # Get current date from the most recent data point
                if len(data) > 0:
                    latest_date = data.index[-1].date()

                    # Find previous trading day's data
                    prev_day_data = data[data.index.date < latest_date]

                    if len(prev_day_data) > 0:
                        # Get the most recent previous day
                        prev_date = prev_day_data.index[-1].date()

                        # Filter data for the previous day
                        prev_day_full = prev_day_data[prev_day_data.index.date == prev_date]

                        if len(prev_day_full) > 0:
                            # Regular Trading Hours mask for MES futures
                            # Typically 9:30 AM to 4:15 PM ET
                            rth_mask = (
                                    ((prev_day_full.index.hour > 9) |
                                     ((prev_day_full.index.hour == 9) & (prev_day_full.index.minute >= 30))) &
                                    ((prev_day_full.index.hour < 16) |
                                     ((prev_day_full.index.hour == 16) & (prev_day_full.index.minute <= 15)))
                            )

                            # Get regular trading hours data
                            rth_data = prev_day_full[rth_mask]

                            # Get extended trading hours data (everything else from that day)
                            eth_data = prev_day_full[~rth_mask]

                            # Add previous day's RTH high and low
                            if len(rth_data) > 0:
                                levels['prev_day_high'] = rth_data['high'].max()
                                levels['prev_day_low'] = rth_data['low'].min()

                            # Add post-hours (ETH) high and low
                            if len(eth_data) > 0:
                                levels['post_hours_high'] = eth_data['high'].max()
                                levels['post_hours_low'] = eth_data['low'].min()

                            # Log levels only once and only if data is found
                            if should_log:
                                print(
                                    f"Previous day RTH levels - High: {levels.get('prev_day_high', 'N/A')}, Low: {levels.get('prev_day_low', 'N/A')}")
                                print(
                                    f"Previous day ETH levels - High: {levels.get('post_hours_high', 'N/A')}, Low: {levels.get('post_hours_low', 'N/A')}")

                                # Set flag to prevent repeated logging
                                self._levels_logged = True

            except Exception as e:
                print(f"Error calculating key levels: {e}")
                # Fallback to a simple method if complex calculation fails
                first_section = data.iloc[:int(len(data) * 0.2)]
                levels['prev_day_high'] = first_section['high'].max()
                levels['prev_day_low'] = first_section['low'].min()

        # If no levels found or not a datetime index, use fallback method
        if not levels:
            first_section = data.iloc[:int(len(data) * 0.2)]
            levels['prev_day_high'] = first_section['high'].max()
            levels['prev_day_low'] = first_section['low'].min()

        # Cache the levels
        self._key_levels_cache = levels

        return levels

    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns in the data.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            pd.DataFrame: Data with added pattern signals.
        """
        # Check if patterns already detected
        data_key = id(data)
        if data_key in self._pattern_cache:
            return self._pattern_cache[data_key]

        df = data.copy()

        # Initialize pattern signal columns
        df['bullish_pattern'] = 0
        df['bearish_pattern'] = 0

        # Detect bullish patterns
        for pattern in self.bullish_patterns:
            try:
                signals = CandlestickPatternDetector.detect_pattern(df, pattern)
                # Add to bullish pattern signals (1 for bullish)
                df['bullish_pattern'] = np.where(signals > 0, 1, df['bullish_pattern'])
            except Exception as e:
                print(f"Error detecting {pattern}: {e}")

        # Detect bearish patterns
        for pattern in self.bearish_patterns:
            try:
                signals = CandlestickPatternDetector.detect_pattern(df, pattern)
                # Add to bearish pattern signals (-1 for bearish)
                df['bearish_pattern'] = np.where(signals < 0, 1, df['bearish_pattern'])
            except Exception as e:
                print(f"Error detecting {pattern}: {e}")

        # Store in cache
        self._pattern_cache[data_key] = df

        return df

    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        # Debug print to show which parameters are being used
        print("\n--- Generating Signals ---")
        print("Input Parameters:")
        for key, value in parameters.items():
            print(f"{key}: {value}")

        # Update parameters
        if parameters is not None:
            self.parameters.update(parameters)

        # Debug: Check if parameters were actually updated
        print("\nCurrent Strategy Parameters:")
        for key, value in self.parameters.items():
            print(f"{key}: {value}")

        # Add more comprehensive indicators with detailed logging
        try:
            print("\nAdding Indicators...")
            df = self._add_indicators(data)

            print("\nDetecting Candlestick Patterns...")
            df = self._detect_candlestick_patterns(df)
        except Exception as e:
            print(f"Error adding indicators or detecting patterns: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(len(data))

        # Initialize signals array
        signals = np.zeros(len(df))

        # Diagnostic logging for conditions
        condition_logs = []

        print("\nGenerating Signals...")
        try:
            for i in range(max(self.parameters['bb_period'], self.parameters['atr_period']) + 1, len(df)):
                # Comprehensive long entry condition diagnostic
                long_conditions = {
                    'below_lower_band': df['close'].iloc[i] < df['bb_lower'].iloc[i],
                    'below_vwap': df['close'].iloc[i] < df['vwap'].iloc[i],
                    'bullish_pattern_detected': df['bullish_pattern'].iloc[i] == 1,
                    'rsi_oversold': df['rsi'].iloc[i] <= self.parameters['rsi_oversold'],
                    'atr_in_range': (self.parameters['atr_min_threshold'] <= df['atr_ratio'].iloc[i] <=
                                     self.parameters['atr_max_threshold']),
                    'bullish_candle': df['close'].iloc[i] > df['open'].iloc[i],
                    'time_details': {
                        'hour': df.index[i].hour if hasattr(df.index, 'hour') else 'N/A',
                        'minute': df.index[i].minute if hasattr(df.index, 'minute') else 'N/A'
                    }
                }

                # Similar detailed short conditions
                short_conditions = {
                    'above_upper_band': df['close'].iloc[i] > df['bb_upper'].iloc[i],
                    'above_vwap': df['close'].iloc[i] > df['vwap'].iloc[i],
                    'bearish_pattern_detected': df['bearish_pattern'].iloc[i] == 1,
                    'rsi_overbought': df['rsi'].iloc[i] >= self.parameters['rsi_overbought'],
                    'atr_in_range': (self.parameters['atr_min_threshold'] <= df['atr_ratio'].iloc[i] <=
                                     self.parameters['atr_max_threshold']),
                    'bearish_candle': df['close'].iloc[i] < df['open'].iloc[i],
                    'time_details': {
                        'hour': df.index[i].hour if hasattr(df.index, 'hour') else 'N/A',
                        'minute': df.index[i].minute if hasattr(df.index, 'minute') else 'N/A'
                    }
                }

                # Count met conditions
                long_met_conditions = sum(1 for v in long_conditions.values() if isinstance(v, bool) and v)
                short_met_conditions = sum(1 for v in short_conditions.values() if isinstance(v, bool) and v)

                # Signal generation with comprehensive logging
                if long_met_conditions >= 4:
                    signals[i] = 1
                    condition_logs.append(('LONG', long_conditions))
                elif short_met_conditions >= 4:
                    signals[i] = -1
                    condition_logs.append(('SHORT', short_conditions))

            # Print detailed condition logs
            print("\nSignal Generation Diagnostic:")
            for signal_type, conditions in condition_logs:
                print(f"\n{signal_type} Signal:")
                for key, value in conditions.items():
                    if key != 'time_details':
                        print(f"  {key}: {value}")

            print(f"\nTotal Signals Generated: {np.sum(signals != 0)}")
            return signals

        except Exception as e:
            print(f"Error during signal generation: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(len(data))

    def optimize(self, data: pd.DataFrame, objective_func: callable,
                 parameter_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], float]:
        import time
        import sys
        import numpy as np
        from itertools import product

        print("\nStarting Detailed Optimization Diagnostics")
        print("-" * 40)

        # Debug: Print the parameter_grid to ensure it's not empty
        print("\nFull Parameter Grid:")
        print(parameter_grid)

        # Explicitly print out the parameter grid to verify
        print("\nParameter Grid Details:")
        grid_details = {
            'bb_period': parameter_grid.get("bb_period", [20]),
            'bb_std': parameter_grid.get("bb_std", [2.0]),
            'atr_period': parameter_grid.get("atr_period", [14]),
            'stop_atr_multiplier': parameter_grid.get("stop_atr_multiplier", [1.5]),
            'rsi_oversold': parameter_grid.get("rsi_oversold", [30]),
            'rsi_overbought': parameter_grid.get("rsi_overbought", [70]),
            'atr_min_threshold': parameter_grid.get("atr_min_threshold", [0.8]),
            'atr_max_threshold': parameter_grid.get("atr_max_threshold", [2.0])
        }

        for key, values in grid_details.items():
            print(f"{key}: {values}")

        # Validate parameter grid
        for key, values in grid_details.items():
            if not values:
                print(f"WARNING: No values for {key}. Using default.")
                grid_details[key] = [self.parameters.get(key, 20)]  # Provide a default

        # Calculate total parameter combinations
        total_combinations = 1
        for values in grid_details.values():
            total_combinations *= len(values)

        print(f"\nTotal parameter combinations to test: {total_combinations}")

        # Initialize timing and tracking variables
        start_time = time.time()
        current_combo = 0
        best_value = -np.inf if objective_func.__name__ != "drawdown" else np.inf
        best_params = None

        # Print a header for detailed tracking
        print("\n{:<10} {:<20} {:<15} {:<15}".format("Progress", "Parameters", "Objective", "Elapsed Time"))
        print("-" * 60)

        # Nested loops for parameter combinations
        try:
            for bb_period in grid_details['bb_period']:
                for bb_std in grid_details['bb_std']:
                    for atr_period in grid_details['atr_period']:
                        for stop_multiplier in grid_details['stop_atr_multiplier']:
                            for rsi_oversold in grid_details['rsi_oversold']:
                                for rsi_overbought in grid_details['rsi_overbought']:
                                    for atr_min in grid_details['atr_min_threshold']:
                                        for atr_max in grid_details['atr_max_threshold']:
                                            # Debug: Print current combination
                                            print(f"\nCurrent Combination: \n{bb_period}, {bb_std}, {atr_period}, "
                                                  f"{stop_multiplier}, {rsi_oversold}, {rsi_overbought}, "
                                                  f"{atr_min}, {atr_max}")

                                            # Skip invalid combinations
                                            if (bb_period <= 0 or bb_std <= 0 or atr_period <= 0 or
                                                    stop_multiplier <= 0 or rsi_oversold >= rsi_overbought or
                                                    atr_min >= atr_max):
                                                print("Skipping invalid combination")
                                                continue

                                            current_combo += 1

                                            # Prepare current parameters
                                            params = {
                                                "bb_period": bb_period,
                                                "bb_std": bb_std,
                                                "atr_period": atr_period,
                                                "stop_atr_multiplier": stop_multiplier,
                                                "rsi_oversold": rsi_oversold,
                                                "rsi_overbought": rsi_overbought,
                                                "atr_min_threshold": atr_min,
                                                "atr_max_threshold": atr_max
                                            }

                                            # Elapsed time calculation with more precision
                                            elapsed_time = time.time() - start_time
                                            progress_percent = (current_combo / total_combinations) * 100

                                            # Debug: Print before signal generation
                                            print("Generating signals...")

                                            # Generate signals
                                            signals = self.generate_signals(data, params)
                                            returns = self.compute_returns(data, signals)

                                            # Calculate objective function value
                                            value = objective_func(returns)

                                            # Print current combination details
                                            print("{:<10.2f} {:<20} {:<15.4f} {:<15.2f}".format(
                                                progress_percent,
                                                f"BB:{bb_period},{bb_std}",
                                                value,
                                                elapsed_time
                                            ))

                                            # Update best parameters
                                            if (objective_func.__name__ != "drawdown" and value > best_value) or \
                                                    (objective_func.__name__ == "drawdown" and value < best_value):
                                                best_value = value
                                                best_params = params.copy()

        except Exception as e:
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()

        # Final results
        print("\nOptimization Complete")
        print(f"Best Parameters: {best_params}")
        print(f"Best {objective_func.__name__}: {best_value}")
        print(f"Total Optimization Time: {time.time() - start_time:.2f} seconds")

        return best_params, best_value

    def compute_returns(self, data: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        """
        Compute strategy returns based on the signals and price data.
        This implementation includes stop loss and target based exits.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray): Array of position signals.

        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Check if we have the required indicators, if not, add them
        if 'atr' not in data.columns:
            data = self._add_indicators(data)

        # Get key levels for targets
        key_levels = self._get_key_levels(data)

        # Initialize returns array
        returns = np.zeros(len(data))

        # Variables to track current position
        in_position = False
        entry_price = 0
        entry_index = 0
        position_type = 0  # 1 for long, -1 for short
        stop_loss = 0
        target = 0

        # Process each bar
        for i in range(1, len(data)):
            # Skip if we don't have ATR
            if np.isnan(data['atr'].iloc[i]):
                continue

            if not in_position:
                # Check for new entry signal
                if signals[i] != 0:
                    in_position = True
                    position_type = signals[i]
                    entry_price = data['close'].iloc[i]
                    entry_index = i

                    # Set stop loss
                    stop_loss = entry_price - (
                                position_type * self.parameters['stop_atr_multiplier'] * data['atr'].iloc[i])

                    # Set target based on nearest key level
                    if position_type > 0:  # Long position
                        # Find the nearest resistance level (previous day high or post-hours high)
                        prev_day_high = key_levels.get('prev_day_high')
                        post_hours_high = key_levels.get('post_hours_high')

                        # Determine which is the nearest level above current price
                        if prev_day_high is not None and post_hours_high is not None:
                            # If both levels exist, use the closer one
                            if abs(prev_day_high - entry_price) <= abs(post_hours_high - entry_price):
                                target = prev_day_high
                            else:
                                target = post_hours_high
                        elif prev_day_high is not None:
                            target = prev_day_high
                        elif post_hours_high is not None:
                            target = post_hours_high
                        else:
                            # Default if no levels available
                            target = entry_price * 1.02
                    else:  # Short position
                        # Find the nearest support level (previous day low or post-hours low)
                        prev_day_low = key_levels.get('prev_day_low')
                        post_hours_low = key_levels.get('post_hours_low')

                        # Determine which is the nearest level below current price
                        if prev_day_low is not None and post_hours_low is not None:
                            # If both levels exist, use the closer one
                            if abs(prev_day_low - entry_price) <= abs(post_hours_low - entry_price):
                                target = prev_day_low
                            else:
                                target = post_hours_low
                        elif prev_day_low is not None:
                            target = prev_day_low
                        elif post_hours_low is not None:
                            target = post_hours_low
                        else:
                            # Default if no levels available
                            target = entry_price * 0.98
            else:
                # Check for exit conditions
                current_price = data['close'].iloc[i]
                exit_signal = False
                exit_reason = None

                # Check stop loss
                if (position_type > 0 and data['low'].iloc[i] <= stop_loss) or \
                        (position_type < 0 and data['high'].iloc[i] >= stop_loss):
                    exit_signal = True
                    exit_reason = "stop_loss"
                    # For returns calculation, assume the exit price is the stop loss
                    exit_price = stop_loss

                # Check target
                elif (position_type > 0 and data['high'].iloc[i] >= target) or \
                        (position_type < 0 and data['low'].iloc[i] <= target):
                    exit_signal = True
                    exit_reason = "target"
                    # For returns calculation, assume the exit price is the target
                    exit_price = target

                # Execute exit if conditions met
                if exit_signal:
                    # Calculate log return for the trade
                    if position_type > 0:  # Long position
                        returns[i] = np.log(exit_price / entry_price)
                    else:  # Short position
                        returns[i] = np.log(entry_price / exit_price)

                    # Reset position tracking variables
                    in_position = False
                    position_type = 0
                    entry_price = 0
                    stop_loss = 0
                    target = 0

        return returns