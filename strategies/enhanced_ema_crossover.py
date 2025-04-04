import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from itertools import product
from tqdm import tqdm


class Strategy:
    def __init__(self, name: str):
        self.name = name


class EnhancedEMACrossoverStrategy(Strategy):
    def __init__(self,
                 ema_period: int = 8,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 volume_multiplier: float = 1.5,
                 max_bars_held: int = 16,
                 bb_window: int = 20,
                 stop_atr_multiplier: float = 1.5,
                 trail_atr_multiplier: float = 2.0,
                 adx_threshold: int = 25):
        super().__init__(name="Enhanced EMA Crossover Strategy")
        self.parameters = {
            "ema_period": ema_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "volume_multiplier": volume_multiplier,
            "max_bars_held": max_bars_held,
            "bb_window": bb_window,
            "stop_atr_multiplier": stop_atr_multiplier,
            "trail_atr_multiplier": trail_atr_multiplier,
            "adx_threshold": adx_threshold
        }
        # Trade tracking variables
        self.trade_count = 0
        self.last_trade_count = 0
        self.last_exits_by_max_bars = 0
        self.last_exits_by_stop_loss = 0
        self.last_exits_by_trailing_stop = 0
        self.last_exits_by_ema = 0
        self.verbose = False
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set a callback function to track optimization progress."""
        self._progress_callback = callback

    def compute_returns(self, data: pd.DataFrame, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns based on signals and price data.

        Args:
            data (pd.DataFrame): Market data with OHLC prices and indicators.
            signals (np.ndarray): Array of position signals.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns array and active positions array.
        """
        data = self._calculate_indicators(data)

        # Calculate log returns (this will be our basis for strategy returns)
        log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0).values

        # Initialize returns and position arrays
        strategy_returns = np.zeros_like(log_returns)
        active_positions = np.zeros_like(log_returns)

        # Variables to track current position
        position = 0
        entry_bar = 0
        entry_price = 0.0
        highest_high = 0.0
        lowest_low = float('inf')

        # Counters for trade statistics
        total_trades = 0
        exits_by_max_bars = 0
        exits_by_stop_loss = 0
        exits_by_trailing_stop = 0
        exits_by_ema = 0

        # Process each bar
        for i in range(1, len(data) - 1):
            # Skip if we don't have indicators
            if i < 20:  # Simple check to make sure indicators are calculated
                continue

            # Update active position tracking
            active_positions[i] = position

            # If we're in a position, check for exit conditions
            if position != 0:
                # For calculating returns
                strategy_returns[i] = position * log_returns[i]

                # Get current EMA
                current_ema = data['ema'].iloc[i]

                # Check for EMA crossover exit
                # We flag the signal when the crossover happens, but execute at the next bar's open

                ema_exit_signal = False
                if (position > 0 and data['close'].iloc[i] < current_ema and
                        data['close'].iloc[i - 1] >= data['ema'].iloc[i - 1]):
                    ema_exit_signal = True
                elif (position < 0 and data['close'].iloc[i] > current_ema and
                      data['close'].iloc[i - 1] <= data['ema'].iloc[i - 1]):
                    ema_exit_signal = True

                # If we have a signal and there's a next bar to execute on
                if ema_exit_signal and i + 1 < len(data):
                    # Calculate the return from current close to next open
                    next_bar_gap = np.log(data['open'].iloc[i + 1] / data['close'].iloc[i])
                    # Add this return when we exit at the next bar's open
                    strategy_returns[i + 1] = position * next_bar_gap
                    position = 0
                    exits_by_ema += 1

                # Update trailing stop logic
                if position > 0:
                    highest_high = max(highest_high, data['close'].iloc[i])
                    trailing_stop = highest_high - self.parameters['trail_atr_multiplier'] * data['atr'].iloc[i]
                    if data['close'].iloc[i] <= trailing_stop:
                        position = 0
                        exits_by_trailing_stop += 1
                elif position < 0:
                    lowest_low = min(lowest_low, data['close'].iloc[i])
                    trailing_stop = lowest_low + self.parameters['trail_atr_multiplier'] * data['atr'].iloc[i]
                    if data['close'].iloc[i] >= trailing_stop:
                        position = 0
                        exits_by_trailing_stop += 1

                # Fixed stop loss check
                if position != 0:
                    atr = data['atr'].iloc[i]
                    stop_loss = entry_price - (position * self.parameters['stop_atr_multiplier'] * atr)
                    if (position > 0 and data['close'].iloc[i] < stop_loss) or \
                            (position < 0 and data['close'].iloc[i] > stop_loss):
                        position = 0
                        exits_by_stop_loss += 1
                    elif i - entry_bar >= self.parameters['max_bars_held']:
                        position = 0
                        exits_by_max_bars += 1

            # Check for new entry signal
            if position == 0 and signals[i] != 0:
                # Check for EMA crossover as the entry trigger
                current_ema = data['ema'].iloc[i]
                prev_ema = data['ema'].iloc[i - 1]

                # Detect EMA crossover
                ema_cross_long = data['close'].iloc[i - 1] <= prev_ema and data['close'].iloc[i] > current_ema
                ema_cross_short = data['close'].iloc[i - 1] >= prev_ema and data['close'].iloc[i] < current_ema

                # Flag entry signals to execute at the next bar's open
                entry_signal = 0
                if signals[i] > 0 and ema_cross_long:
                    entry_signal = 1
                elif signals[i] < 0 and ema_cross_short:
                    entry_signal = -1

                # If we have an entry signal and there's a next bar to execute on
                if entry_signal != 0 and i + 1 < len(data):
                    # The entry happens at the next bar's open
                    position = entry_signal
                    entry_bar = i + 1  # Next bar
                    entry_price = data['open'].iloc[i + 1]

                    if position > 0:
                        highest_high = entry_price
                    else:
                        lowest_low = entry_price

                    total_trades += 1

            # Update active position for the next bar
            active_positions[i + 1] = position

        # Store trade statistics
        self.last_trade_count = total_trades
        self.last_exits_by_max_bars = exits_by_max_bars
        self.last_exits_by_stop_loss = exits_by_stop_loss
        self.last_exits_by_trailing_stop = exits_by_trailing_stop
        self.last_exits_by_ema = exits_by_ema

        # Print trade statistics if verbose
        if self.verbose:
            print("===== Trading Statistics =====")
            print(f"Total trades executed: {total_trades}")
            print(f"Exits by max bars held: {exits_by_max_bars}")
            print(f"Exits by stop loss: {exits_by_stop_loss}")
            print(f"Exits by trailing stop: {exits_by_trailing_stop}")
            print(f"Exits by EMA crossover: {exits_by_ema}")

        return strategy_returns, active_positions

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for this strategy."""
        df = data.copy()

        # Calculate 8 EMA
        df['ema'] = df['close'].ewm(span=self.parameters['ema_period'], adjust=False).mean()

        # Original Bollinger Bands
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

        # Add SMAs for trend context
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()

        # Calculate ADX for trend strength
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

        return df

    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate trading signals based on strategy rules.

        This strategy uses the Enhanced Market Regime logic as a filter and then
        uses 8 EMA crossovers for entries and exits.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters. Defaults to None.

        Returns:
            np.ndarray: Array of position signals (1 for long, -1 for short, 0 for no position).
        """
        if parameters is not None:
            self.parameters.update(parameters)

        # Calculate indicators
        data = self._calculate_indicators(data)

        # Initialize signals array
        signals = np.zeros(len(data))

        p = self.parameters

        # Loop through the data to generate signals
        for i in range(2, len(data)):
            prev_row = data.iloc[i - 1]
            earlier_row = data.iloc[i - 2]

            # Market Regime Filter Logic (same as in EnhancedMarketRegimeStrategy)
            mean_rev_long = (
                    earlier_row['low'] < earlier_row['lower_band'] and
                    earlier_row['RSI'] < p['rsi_oversold'] and
                    earlier_row['volume'] > p['volume_multiplier'] * earlier_row['avg_volume'] and
                    prev_row['open'] > earlier_row['low'] * 1.0005
            )

            mean_rev_short = (
                    earlier_row['high'] > earlier_row['upper_band'] and
                    earlier_row['RSI'] > p['rsi_overbought'] and
                    earlier_row['volume'] > p['volume_multiplier'] * earlier_row['avg_volume'] and
                    prev_row['open'] < earlier_row['high'] * 0.9995
            )

            # Add ADX filter from original strategy
            if earlier_row['adx'] < p['adx_threshold']:
                if mean_rev_long:
                    signals[i] = 1
                elif mean_rev_short:
                    signals[i] = -1

        return signals

    def optimize(self, data: pd.DataFrame, objective_func: callable,
                 parameter_grid: Dict[str, List[Any]], n_jobs: int = -1) -> Tuple[Dict[str, Any], float]:
        """
        Optimize strategy parameters using grid search with parallel processing.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            objective_func (callable): Function to maximize during optimization.
            parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.

        Returns:
            Tuple[Dict[str, Any], float]: Best parameters and best objective value.
        """
        from joblib import Parallel, delayed
        import itertools
        import sys
        import multiprocessing

        best_value = -np.inf if objective_func.__name__ != "drawdown" else np.inf
        best_params = {}

        # Extract parameter values from grid with defaults
        param_keys = parameter_grid.keys()
        param_values = [parameter_grid.get(key, [self.parameters.get(key)]) for key in param_keys]

        # Create all parameter combinations
        param_combinations = list(itertools.product(*param_values))

        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations...")

        # Use a multiprocessing manager for shared counter
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)

        def update_progress(total, counter):
            """Update progress in a way that can be pickled."""
            counter.value += 1
            sys.stdout.write(f"\rProgress: {counter.value}/{total} ({counter.value / total * 100:.2f}%)")
            sys.stdout.flush()

            # Call the external progress callback if provided
            if self._progress_callback:
                self._progress_callback()

        # Define function to evaluate a single parameter set
        def evaluate_params(params_tuple, total_combinations, progress_counter):
            try:
                # Create parameter dictionary
                params = {name: value for name, value in zip(param_keys, params_tuple)}

                # Skip invalid combinations
                if 'rsi_oversold' in params and 'rsi_overbought' in params:
                    if params['rsi_oversold'] >= params['rsi_overbought']:
                        update_progress(total_combinations, progress_counter)
                        return None, -np.inf if objective_func.__name__ != "drawdown" else np.inf

                # Generate signals with the current parameters
                signals = self.generate_signals(data, params)

                # Compute returns
                returns, _ = self.compute_returns(data, signals)

                # Calculate objective value
                value = objective_func(returns)

                # Update progress
                update_progress(total_combinations, progress_counter)

                return params, value
            except Exception as e:
                print(f"Error with parameters {params_tuple}: {str(e)}")
                update_progress(total_combinations, progress_counter)
                return None, -np.inf if objective_func.__name__ != "drawdown" else np.inf

        # Parallel execution
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_params)(params, total_combinations, counter)
            for params in param_combinations
        )

        # Print a newline after progress updates
        print()

        # Find best results
        valid_results = [r for r in results if r[0] is not None]

        if not valid_results:
            print("No valid parameter combinations found!")
            return self.parameters.copy(), -np.inf

        for params, value in valid_results:
            if (objective_func.__name__ != "drawdown" and value > best_value) or \
                    (objective_func.__name__ == "drawdown" and value < best_value):
                best_value = value
                best_params = params

        print(f"\nBest Parameters: {best_params}")
        print(f"Best {objective_func.__name__}: {best_value}")

        return best_params, best_value