import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional
from itertools import product

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

        if isinstance(data.index, pd.DatetimeIndex):
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
                        # Regular Trading Hours typically 9:30 AM to 4:00 PM ET for US markets
                        # For MES futures, regular session is typically 9:30 AM to 4:15 PM ET
                        rth_mask = ((prev_day_full.index.hour > 9) |
                                    ((prev_day_full.index.hour == 9) & (prev_day_full.index.minute >= 30))) & \
                                   ((prev_day_full.index.hour < 16) |
                                    ((prev_day_full.index.hour == 16) & (prev_day_full.index.minute <= 15)))

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

                        # Only log once to avoid too many repeated logs
                        if self._key_levels_cache is None:
                            print(
                                f"Previous day RTH levels - High: {levels.get('prev_day_high', 'N/A')}, Low: {levels.get('prev_day_low', 'N/A')}")
                            print(
                                f"Previous day ETH levels - High: {levels.get('post_hours_high', 'N/A')}, Low: {levels.get('post_hours_low', 'N/A')}")

            # If we couldn't get the specific levels, fall back to daily resampling
            if not levels:
                print("Using daily resampling fallback for key levels...")
                daily_data = data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                }).dropna()

                if len(daily_data) > 1:
                    # Get previous day's high and low
                    prev_day = daily_data.iloc[-2]
                    levels['prev_day_high'] = prev_day['high']
                    levels['prev_day_low'] = prev_day['low']
        else:
            # If no datetime information, use simple approach
            # Use highest high and lowest low from first 20% of data as key levels
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
        """
        Generate trading signals based on Bollinger Bands, VWAP, and candlestick patterns.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.

        Returns:
            np.ndarray: Array of position signals (1 for long, 0 for flat, -1 for short).
        """
        if parameters is not None:
            # If parameters change, clear caches
            if parameters != self.parameters:
                self._indicator_cache = {}
                self._pattern_cache = {}
                self._key_levels_cache = None

            self.parameters.update(parameters)

        # Add indicators
        df = self._add_indicators(data)

        # Get key levels (only once)
        key_levels = self._get_key_levels(data)

        # Detect candlestick patterns
        df = self._detect_candlestick_patterns(df)

        # Initialize signals array
        signals = np.zeros(len(df))

        # Generate entry signals
        for i in range(max(self.parameters['bb_period'], self.parameters['atr_period']) + 1, len(df)):
            # Skip if not in morning session
            if not df['morning_session'].iloc[i]:
                continue

            # Check for long entry conditions
            if (df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                    df['close'].iloc[i] < df['vwap'].iloc[i] and
                    df['open'].iloc[i] < df['vwap'].iloc[i] and
                    df['bullish_pattern'].iloc[i] == 1 and
                    # RSI filter - ensure RSI is in oversold territory for long entries
                    df['rsi'].iloc[i] <= self.parameters['rsi_oversold'] and
                    # ATR filter - ensure volatility is within acceptable range
                    df['atr_ratio'].iloc[i] >= self.parameters['atr_min_threshold'] and
                    df['atr_ratio'].iloc[i] <= self.parameters['atr_max_threshold']):
                signals[i] = 1  # Long signal

            # Check for short entry conditions
            elif (df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                  df['close'].iloc[i] > df['vwap'].iloc[i] and
                  df['open'].iloc[i] > df['vwap'].iloc[i] and
                  df['bearish_pattern'].iloc[i] == 1 and
                  # RSI filter - ensure RSI is in overbought territory for short entries
                  df['rsi'].iloc[i] >= self.parameters['rsi_overbought'] and
                  # ATR filter - ensure volatility is within acceptable range
                  df['atr_ratio'].iloc[i] >= self.parameters['atr_min_threshold'] and
                  df['atr_ratio'].iloc[i] <= self.parameters['atr_max_threshold']):
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

        # Clear caches before optimization
        self._indicator_cache = {}
        self._pattern_cache = {}
        self._key_levels_cache = None

        # Extract parameter lists from grid
        bb_periods = parameter_grid.get("bb_period", [self.parameters["bb_period"]])
        bb_stds = parameter_grid.get("bb_std", [self.parameters["bb_std"]])
        atr_periods = parameter_grid.get("atr_period", [self.parameters["atr_period"]])
        stop_multipliers = parameter_grid.get("stop_atr_multiplier", [self.parameters["stop_atr_multiplier"]])
        rsi_oversolds = parameter_grid.get("rsi_oversold", [self.parameters["rsi_oversold"]])
        rsi_overboughts = parameter_grid.get("rsi_overbought", [self.parameters["rsi_overbought"]])
        atr_min_thresholds = parameter_grid.get("atr_min_threshold", [self.parameters["atr_min_threshold"]])
        atr_max_thresholds = parameter_grid.get("atr_max_threshold", [self.parameters["atr_max_threshold"]])

        # Calculate total parameter combinations
        total_combinations = (
                len(bb_periods) * len(bb_stds) * len(atr_periods) * len(stop_multipliers) *
                len(rsi_oversolds) * len(rsi_overboughts) * len(atr_min_thresholds) * len(atr_max_thresholds)
        )

        print(f"Starting optimization with {total_combinations} parameter combinations")
        start_time = time.time()
        current_combo = 0

        # Iterate through all parameter combinations
        for bb_period, bb_std in product(bb_periods, bb_stds):
            for atr_period, stop_multiplier in product(atr_periods, stop_multipliers):
                for rsi_oversold, rsi_overbought in product(rsi_oversolds, rsi_overboughts):
                    for atr_min, atr_max in product(atr_min_thresholds, atr_max_thresholds):
                        # Skip invalid combinations
                        if (bb_period <= 0 or bb_std <= 0 or atr_period <= 0 or stop_multiplier <= 0 or
                                rsi_oversold >= rsi_overbought or atr_min >= atr_max):
                            continue

                        current_combo += 1
                        if current_combo % 10 == 0:  # Report progress every 10 combinations
                            elapsed = time.time() - start_time
                            eta = (elapsed / current_combo) * (total_combinations - current_combo)
                            print(
                                f"Testing combination {current_combo}/{total_combinations} - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

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

                        # Generate signals with current parameters
                        signals = self.generate_signals(data, params)

                        # Calculate returns
                        returns = self.compute_returns(data, signals)

                        # Calculate objective function value
                        value = objective_func(returns)

                        # Update best parameters if needed
                        if (objective_func.__name__ != "drawdown" and value > best_value) or \
                                (objective_func.__name__ == "drawdown" and value < best_value):
                            best_value = value
                            best_params = params.copy()
                            print(
                                f"New best {objective_func.__name__}: {best_value:.4f} with parameters: {best_params}")

        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.1f} seconds")
        print(f"Best {objective_func.__name__}: {best_value:.4f}")
        print(f"Best parameters: {best_params}")

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