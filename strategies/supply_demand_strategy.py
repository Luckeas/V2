import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from strategies.base import Strategy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


class SupplyDemandStrategy(Strategy):
    """
    Supply and Demand Strategy based on the 3-step formula:
    1. Market Structure Analysis: Identify uptrend or downtrend
    2. Supply and Demand Zones: Buy at demand zones in uptrends, sell at supply zones in downtrends
    3. Risk-to-Reward Filter: Only take trades with R:R ratio >= 2.5:1
    """

    def __init__(self,
                 consolidation_lookback: int = 3,
                 consolidation_threshold: float = 0.3,
                 min_risk_reward_ratio: float = 2.5,
                 stop_loss_buffer: float = 0.05,
                 max_bars_held: int = 20):
        """
        Initialize the Supply and Demand Strategy.

        Args:
            consolidation_lookback (int): Number of bars to determine consolidation
            consolidation_threshold (float): Max % range for consolidation candles
            min_risk_reward_ratio (float): Minimum risk-reward ratio to take a trade (e.g. 2.5)
            stop_loss_buffer (float): Additional buffer for stop loss as % of zone height
            max_bars_held (int): Maximum number of bars to hold a position
        """
        super().__init__(name="Supply and Demand Strategy")
        self.parameters = {
            "consolidation_lookback": consolidation_lookback,
            "consolidation_threshold": consolidation_threshold,
            "min_risk_reward_ratio": min_risk_reward_ratio,
            "stop_loss_buffer": stop_loss_buffer,
            "max_bars_held": max_bars_held
        }

        # Initialize tracking variables for trades
        self.trade_count = 0
        self.last_trade_count = 0
        self.last_exits_by_max_bars = 0
        self.last_exits_by_stop_loss = 0
        self.last_exits_by_target = 0

        # For progress tracking during optimization
        self._progress_callback = None

        # Debug mode for additional outputs
        self.debug = False

    def set_progress_callback(self, callback):
        """Set a callback function to track optimization progress."""
        self._progress_callback = callback

    def _identify_market_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market structure (uptrend or downtrend) based on higher highs/lows
        or lower highs/lows and add columns to the dataframe.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            pd.DataFrame: Data with added columns for market structure analysis.
        """
        df = data.copy()

        # Initialize columns
        df['trend'] = 0  # 1 for uptrend, -1 for downtrend, 0 for undefined
        df['valid_high'] = np.nan
        df['valid_low'] = np.nan
        df['last_valid_high'] = np.nan
        df['last_valid_low'] = np.nan

        # Need at least 5 bars to determine structure
        if len(df) < 5:
            return df

        # Initialize first valid high and low indices with numeric index positions
        highest_point_idx = df['high'].iloc[:3].idxmax()
        lowest_point_idx = df['low'].iloc[:3].idxmin()

        df.loc[highest_point_idx, 'valid_high'] = df.loc[highest_point_idx, 'high']
        df.loc[lowest_point_idx, 'valid_low'] = df.loc[lowest_point_idx, 'low']

        # Get the numeric indices for comparison
        highest_point_pos = df.index.get_loc(highest_point_idx)
        lowest_point_pos = df.index.get_loc(lowest_point_idx)

        # Start with undefined trend
        current_trend = 0

        # Last confirmed valid high and low
        last_valid_high = df.loc[highest_point_idx, 'high']
        last_valid_low = df.loc[lowest_point_idx, 'low']
        last_valid_high_pos = highest_point_pos
        last_valid_low_pos = lowest_point_pos

        # Process each bar from index 3 onwards
        for i in range(3, len(df)):
            current_idx = df.index[i]  # Get the actual index (timestamp)

            # Initialize with previous trend
            df.loc[current_idx, 'trend'] = current_trend

            # Reference to the current row
            current_high = df.loc[current_idx, 'high']
            current_low = df.loc[current_idx, 'low']

            # Update last valid high and low
            df.loc[current_idx, 'last_valid_high'] = last_valid_high
            df.loc[current_idx, 'last_valid_low'] = last_valid_low

            # Check for new valid high (ONLY if we've made a valid low after the last valid high)
            if current_high > last_valid_high and last_valid_high_pos < last_valid_low_pos:
                df.loc[current_idx, 'valid_high'] = current_high
                last_valid_high = current_high
                last_valid_high_pos = i  # Use numeric position

                # If we're making higher highs and we already have valid lows, we're in an uptrend
                if last_valid_low_pos > 0 and current_trend != 1:
                    current_trend = 1
                    df.loc[current_idx, 'trend'] = current_trend

            # Check for new valid low (ONLY if we've made a valid high after the last valid low)
            if current_low < last_valid_low and last_valid_low_pos < last_valid_high_pos:
                df.loc[current_idx, 'valid_low'] = current_low
                last_valid_low = current_low
                last_valid_low_pos = i  # Use numeric position

                # If we're making lower lows and we already have valid highs, we're in a downtrend
                if last_valid_high_pos > 0 and current_trend != -1:
                    current_trend = -1
                    df.loc[current_idx, 'trend'] = current_trend

        return df

    def _identify_consolidation_areas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify consolidation areas and add columns for supply and demand zones.

        Args:
            data (pd.DataFrame): Market data with market structure analysis.

        Returns:
            pd.DataFrame: Data with added columns for supply and demand zones.
        """
        df = data.copy()

        # Initialize columns
        df['is_consolidation'] = False
        df['is_impulse'] = False
        df['supply_zone_start'] = np.nan
        df['supply_zone_end'] = np.nan
        df['demand_zone_start'] = np.nan
        df['demand_zone_end'] = np.nan

        consolidation_lookback = self.parameters['consolidation_lookback']
        consolidation_threshold = self.parameters['consolidation_threshold']

        # Need at least consolidation_lookback+2 bars
        if len(df) < consolidation_lookback + 2:
            return df

        # Find consolidation areas
        for i in range(consolidation_lookback, len(df) - 1):
            current_idx = df.index[i]  # Current timestamp index
            next_idx = df.index[i + 1]  # Next timestamp index

            # Calculate the range of the lookback period
            lookback_slice = df.iloc[i - consolidation_lookback:i]
            range_percent = (lookback_slice['high'].max() - lookback_slice['low'].min()) / lookback_slice['low'].min()

            # Check if the range is below the threshold (consolidation)
            if range_percent < consolidation_threshold:
                df.loc[current_idx, 'is_consolidation'] = True

                # Check if the next candle is an impulse move
                current_range = (df.loc[current_idx, 'high'] - df.loc[current_idx, 'low']) / df.loc[current_idx, 'low']
                next_range = (df.loc[next_idx, 'high'] - df.loc[next_idx, 'low']) / df.loc[next_idx, 'low']
                next_move = df.loc[next_idx, 'close'] - df.loc[current_idx, 'close']

                # Define impulse move: significantly larger range and directional
                if next_range > current_range * 1.5:
                    df.loc[next_idx, 'is_impulse'] = True

                    # Check direction of impulse move
                    if next_move > 0:  # Bullish impulse
                        # Mark this as a demand zone
                        df.loc[current_idx, 'demand_zone_start'] = df.loc[current_idx, 'low']
                        df.loc[current_idx, 'demand_zone_end'] = df.loc[current_idx, 'high']
                    else:  # Bearish impulse
                        # Mark this as a supply zone
                        df.loc[current_idx, 'supply_zone_start'] = df.loc[current_idx, 'low']
                        df.loc[current_idx, 'supply_zone_end'] = df.loc[current_idx, 'high']

        # DEBUG: Count consolidation areas and impulse moves
        consolidation_count = df['is_consolidation'].sum()
        impulse_count = df['is_impulse'].sum()

        print(f"Found {consolidation_count} consolidation areas and {impulse_count} impulse moves")

        return df

    def _calculate_risk_reward(self, entry_price: float, stop_price: float, target_price: float) -> float:
        """
        Calculate risk-to-reward ratio for a trade.

        Args:
            entry_price (float): Entry price
            stop_price (float): Stop loss price
            target_price (float): Take profit price

        Returns:
            float: Risk-to-reward ratio
        """
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)

        if risk == 0:
            return 0

        return reward / risk

    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate trading signals based on the supply and demand strategy.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            parameters (Dict[str, Any], optional): Strategy parameters to override defaults.

        Returns:
            np.ndarray: Array of trading signals (1 for long, -1 for short, 0 for no position)
        """
        if parameters is not None:
            self.parameters.update(parameters)

        # Initialize signals array
        signals = np.zeros(len(data))

        # Process market structure and find zones
        df = self._identify_market_structure(data)
        df = self._identify_consolidation_areas(df)

        # DEBUG: Count trends and zones
        trend_counts = {-1: 0, 0: 0, 1: 0}
        zone_counts = {'supply': 0, 'demand': 0}

        # Count trends in the data
        for i in range(len(df)):
            trend_counts[df.iloc[i]['trend']] += 1

        # Count zones
        zone_counts['supply'] = df['supply_zone_start'].notna().sum()
        zone_counts['demand'] = df['demand_zone_start'].notna().sum()

        print(f"Trend counts: {trend_counts}")
        print(f"Zone counts: {zone_counts}")

        # Reset trade statistics
        self.trade_count = 0
        self.last_exits_by_max_bars = 0
        self.last_exits_by_stop_loss = 0
        self.last_exits_by_target = 0

        # Variables to track position state
        in_position = False
        entry_bar = 0
        position_type = 0

        # Find recent confirmed highs and lows for take profit targets
        recent_high = df['high'].iloc[0]
        recent_low = df['low'].iloc[0]

        # Loop through each bar to generate signals
        for i in range(1, len(df)):
            current_idx = df.index[i]  # Get actual timestamp index

            # Skip if we're already in a position
            if in_position:
                # Track position for the required number of bars
                if i - entry_bar >= self.parameters['max_bars_held']:
                    signals[i] = 0  # Exit position
                    in_position = False
                    self.last_exits_by_max_bars += 1
                continue

            current_price = df.loc[current_idx, 'close']
            current_trend = df.loc[current_idx, 'trend']

            # Only look for entry if we have a definite trend
            if current_trend == 0:
                continue

            # Update recent high/low for take profit targets
            if i > 10:
                recent_high = df['high'].iloc[i - 10:i].max()
                recent_low = df['low'].iloc[i - 10:i].min()

            # In uptrend, look for demand zones
            if current_trend == 1:
                # Check if we're in a demand zone
                for j in range(max(0, i - 20), i):
                    look_idx = df.index[j]  # Get actual timestamp index
                    zone_start = df.loc[look_idx, 'demand_zone_start']
                    zone_end = df.loc[look_idx, 'demand_zone_end']

                    if pd.notna(zone_start) and pd.notna(zone_end):
                        # Check if current price is within the demand zone
                        if zone_start <= current_price <= zone_end:
                            # Calculate stop loss and take profit
                            stop_loss = zone_start * (1 - self.parameters['stop_loss_buffer'])
                            take_profit = recent_high

                            # Calculate risk-reward ratio
                            rr_ratio = self._calculate_risk_reward(current_price, stop_loss, take_profit)

                            # Only take trade if R:R meets minimum requirement
                            if rr_ratio >= self.parameters['min_risk_reward_ratio']:
                                signals[i] = 1  # Long signal
                                in_position = True
                                entry_bar = i
                                position_type = 1
                                self.trade_count += 1
                                break

            # In downtrend, look for supply zones
            elif current_trend == -1:
                # Check if we're in a supply zone
                for j in range(max(0, i - 20), i):
                    look_idx = df.index[j]  # Get actual timestamp index
                    zone_start = df.loc[look_idx, 'supply_zone_start']
                    zone_end = df.loc[look_idx, 'supply_zone_end']

                    if pd.notna(zone_start) and pd.notna(zone_end):
                        # Check if current price is within the supply zone
                        if zone_start <= current_price <= zone_end:
                            # Calculate stop loss and take profit
                            stop_loss = zone_end * (1 + self.parameters['stop_loss_buffer'])
                            take_profit = recent_low

                            # Calculate risk-reward ratio
                            rr_ratio = self._calculate_risk_reward(current_price, stop_loss, take_profit)

                            # Only take trade if R:R meets minimum requirement
                            if rr_ratio >= self.parameters['min_risk_reward_ratio']:
                                signals[i] = -1  # Short signal
                                in_position = True
                                entry_bar = i
                                position_type = -1
                                self.trade_count += 1
                                break

        # Save trade count for reporting
        self.last_trade_count = self.trade_count

        # Add this in generate_signals() method, just before the return statement
        print(f"Data range: {df.index[0]} to {df.index[-1]}, total bars: {len(df)}")

        # Counter for entry conditions
        total_zone_checks = 0
        price_in_zone_count = 0
        rr_insufficient_count = 0
        potential_entries = 0

        # Loop through debug
        for i in range(1, len(df)):
            current_idx = df.index[i]
            current_price = df.loc[current_idx, 'close']
            current_trend = df.loc[current_idx, 'trend']

            # Only count when we have a definite trend
            if current_trend == 0:
                continue

            # Sample check for one condition (adjust to match your actual logic)
            if current_trend == 1:  # Uptrend
                for j in range(max(0, i - 20), i):
                    look_idx = df.index[j]
                    zone_start = df.loc[look_idx, 'demand_zone_start']
                    zone_end = df.loc[look_idx, 'demand_zone_end']

                    if pd.notna(zone_start) and pd.notna(zone_end):
                        total_zone_checks += 1

                        # Check if price is in zone
                        if zone_start <= current_price <= zone_end:
                            price_in_zone_count += 1

                            # Calculate stop loss and take profit (simplified)
                            stop_loss = zone_start * (1 - self.parameters['stop_loss_buffer'])
                            recent_high = df['high'].iloc[max(0, i - 10):i].max()
                            take_profit = recent_high

                            # Calculate risk-reward ratio
                            risk = abs(current_price - stop_loss)
                            reward = abs(take_profit - current_price)

                            if risk > 0:
                                rr_ratio = reward / risk
                                # Check if R:R meets minimum requirement
                                if rr_ratio >= self.parameters['min_risk_reward_ratio']:
                                    potential_entries += 1
                                else:
                                    rr_insufficient_count += 1

        print(f"Entry condition stats:")
        print(f"  Total zone checks: {total_zone_checks}")
        print(f"  Price in zone count: {price_in_zone_count}")
        print(f"  R:R insufficient count: {rr_insufficient_count}")
        print(f"  Potential entries: {potential_entries}")

        return signals

    def compute_returns(self, data: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        """
        Compute returns based on signals, handling stop losses and take profits.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray): Array of trading signals.

        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Run market structure and zone identification for stop loss and take profit levels
        df = self._identify_market_structure(data)
        df = self._identify_consolidation_areas(df)

        # Initialize returns array
        returns = np.zeros(len(data))

        # Initialize position state
        position = 0
        entry_price = 0
        entry_bar = 0
        stop_loss = 0
        take_profit = 0

        # Reset trade statistics
        self.last_exits_by_max_bars = 0
        self.last_exits_by_stop_loss = 0
        self.last_exits_by_target = 0

        # Track active positions for trade counts
        active_positions = np.zeros(len(data))

        # Process signals
        for i in range(1, len(data)):
            current_idx = data.index[i]  # Get actual timestamp index

            if position != 0:
                # Calculate return for the current bar
                log_return = np.log(data.loc[current_idx, 'close'] / data.iloc[i - 1]['close'])
                returns[i] = position * log_return
                active_positions[i] = position

                # Check if stop loss was hit
                if position > 0 and data.loc[current_idx, 'low'] <= stop_loss:
                    # Calculate actual return with slippage to stop loss
                    stop_return = np.log(stop_loss / data.iloc[i - 1]['close'])
                    returns[i] = position * stop_return
                    position = 0
                    self.last_exits_by_stop_loss += 1
                elif position < 0 and data.loc[current_idx, 'high'] >= stop_loss:
                    # Calculate actual return with slippage to stop loss
                    stop_return = np.log(stop_loss / data.iloc[i - 1]['close'])
                    returns[i] = position * stop_return
                    position = 0
                    self.last_exits_by_stop_loss += 1

                # Check if take profit was hit
                elif position > 0 and data.loc[current_idx, 'high'] >= take_profit:
                    # Calculate actual return with slippage to take profit
                    tp_return = np.log(take_profit / data.iloc[i - 1]['close'])
                    returns[i] = position * tp_return
                    position = 0
                    self.last_exits_by_target += 1
                elif position < 0 and data.loc[current_idx, 'low'] <= take_profit:
                    # Calculate actual return with slippage to take profit
                    tp_return = np.log(take_profit / data.iloc[i - 1]['close'])
                    returns[i] = position * tp_return
                    position = 0
                    self.last_exits_by_target += 1

                # Check if max bars held is reached
                elif i - entry_bar >= self.parameters['max_bars_held']:
                    position = 0
                    self.last_exits_by_max_bars += 1

            # Check for new signal
            if position == 0 and signals[i] != 0:
                position = signals[i]
                entry_price = data.loc[current_idx, 'close']
                entry_bar = i
                active_positions[i] = position

                # Set default stop loss and take profit
                stop_loss = entry_price * (1 - 0.01) if position > 0 else entry_price * (1 + 0.01)
                take_profit = entry_price * (1 + 0.02) if position > 0 else entry_price * (1 - 0.02)

                # Find the most recent demand/supply zone for stop loss
                window_start = max(0, i - 20)
                if position > 0:  # Long position
                    # Look for demand zone
                    for j in range(i - 1, window_start - 1, -1):
                        look_idx = df.index[j]  # Get actual timestamp index
                        if pd.notna(df.loc[look_idx, 'demand_zone_start']):
                            zone_start = df.loc[look_idx, 'demand_zone_start']
                            stop_loss = zone_start * (1 - self.parameters['stop_loss_buffer'])
                            break

                    # Set take profit at recent high
                    recent_high = data['high'].iloc[max(0, i - 10):i].max()
                    take_profit = recent_high
                else:  # Short position
                    # Look for supply zone
                    for j in range(i - 1, window_start - 1, -1):
                        look_idx = df.index[j]  # Get actual timestamp index
                        if pd.notna(df.loc[look_idx, 'supply_zone_end']):
                            zone_end = df.loc[look_idx, 'supply_zone_end']
                            stop_loss = zone_end * (1 + self.parameters['stop_loss_buffer'])
                            break

                    # Set take profit at recent low
                    recent_low = data['low'].iloc[max(0, i - 10):i].min()
                    take_profit = recent_low

        # Store active_positions for reference but only return the returns array
        self.active_positions = active_positions
        return returns

    def optimize(self, data: pd.DataFrame, objective_func: callable,
                 parameter_grid: Dict[str, List[Any]], n_jobs: int = 8) -> Tuple[Dict[str, Any], float]:
        """
        Optimize strategy parameters using parallel processing.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            objective_func (callable): Function to optimize (e.g., sharpe_ratio).
            parameter_grid (Dict[str, List[Any]]): Grid of parameters to search.
            n_jobs (int): Number of parallel jobs to run.

        Returns:
            Tuple[Dict[str, Any], float]: Best parameters and corresponding objective value.
        """
        import itertools
        from tqdm import tqdm

        keys, values = zip(*parameter_grid.items())
        combinations = list(itertools.product(*values))
        total_combinations = len(combinations)

        # Compute these once outside parallel loop
        market_structure_df = self._identify_market_structure(data)
        prepared_df = self._identify_consolidation_areas(market_structure_df)

        if self.debug:
            print(f"Testing {total_combinations} parameter combinations...")

        def evaluate_params(combo):
            params = dict(zip(keys, combo))
            signals = self.generate_signals(prepared_df, params)
            returns = self.compute_returns(prepared_df, signals)
            value = objective_func(returns)
            return params, value

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_params)(combo) for combo in tqdm(combinations)
        )

        # Find best result
        if objective_func.__name__ == "drawdown":
            # For drawdown, lower is better
            best_params, best_value = min(results, key=lambda x: x[1])
        else:
            # For other metrics like Sharpe ratio, higher is better
            best_params, best_value = max(results, key=lambda x: x[1])

        if self.debug:
            print(f"Optimization complete. Best value: {best_value:.4f}")
            print(f"Best parameters: {best_params}")

        return best_params, best_value

    def plot_zones(self, data: pd.DataFrame, window_start: int = 0, window_size: int = 100):
        """
        Plot the market structure and supply/demand zones for visual analysis.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            window_start (int): Starting index for the window to plot.
            window_size (int): Number of bars to show in the plot.
        """
        # Process market structure and find zones
        df = self._identify_market_structure(data)
        df = self._identify_consolidation_areas(df)

        # Create window for plotting
        window_end = min(window_start + window_size, len(df))
        plot_df = df.iloc[window_start:window_end]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot candlesticks
        for i in range(len(plot_df)):
            idx = plot_df.index[i]
            open_price = plot_df.loc[idx, 'open']
            close_price = plot_df.loc[idx, 'close']
            high_price = plot_df.loc[idx, 'high']
            low_price = plot_df.loc[idx, 'low']

            # Determine candle color
            if close_price >= open_price:
                color = 'green'
            else:
                color = 'red'

            # Plot candle body
            body_bottom = min(open_price, close_price)
            body_height = abs(close_price - open_price)
            ax.bar(i, body_height, bottom=body_bottom, width=0.6, color=color, alpha=0.7)

            # Plot candle wicks
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)

        # Mark valid highs and lows
        for i in range(len(plot_df)):
            idx = plot_df.index[i]

            if pd.notna(plot_df.loc[idx, 'valid_high']):
                ax.scatter(i, plot_df.loc[idx, 'valid_high'], color='blue', marker='^', s=100)
                ax.text(i, plot_df.loc[idx, 'valid_high'] * 1.002, 'VH', ha='center')

            if pd.notna(plot_df.loc[idx, 'valid_low']):
                ax.scatter(i, plot_df.loc[idx, 'valid_low'], color='blue', marker='v', s=100)
                ax.text(i, plot_df.loc[idx, 'valid_low'] * 0.998, 'VL', ha='center')

        # Mark supply and demand zones
        for i in range(len(plot_df)):
            idx = plot_df.index[i]

            if pd.notna(plot_df.loc[idx, 'supply_zone_start']) and pd.notna(plot_df.loc[idx, 'supply_zone_end']):
                zone_start = plot_df.loc[idx, 'supply_zone_start']
                zone_end = plot_df.loc[idx, 'supply_zone_end']
                ax.axhspan(zone_start, zone_end, alpha=0.2, color='red', linestyle='--')
                ax.text(i, zone_end, 'S', color='red', ha='center')

            if pd.notna(plot_df.loc[idx, 'demand_zone_start']) and pd.notna(plot_df.loc[idx, 'demand_zone_end']):
                zone_start = plot_df.loc[idx, 'demand_zone_start']
                zone_end = plot_df.loc[idx, 'demand_zone_end']
                ax.axhspan(zone_start, zone_end, alpha=0.2, color='green', linestyle='--')
                ax.text(i, zone_start, 'D', color='green', ha='center')

        # Highlight trend
        trend_colors = {0: 'gray', 1: 'green', -1: 'red'}
        for i in range(len(plot_df)):
            trend = plot_df.iloc[i]['trend']
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color=trend_colors[trend])

        # Set labels and title
        ax.set_title('Supply and Demand Strategy Analysis')
        ax.set_xlabel('Bar Number')
        ax.set_ylabel('Price')

        # Set x-axis ticks
        xticks = list(range(0, len(plot_df), 10))
        ax.set_xticks(xticks)
        ax.set_xticklabels([plot_df.index[i].strftime('%Y-%m-%d %H:%M') if i < len(plot_df) else '' for i in xticks])
        plt.xticks(rotation=45)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.2, label='Demand Zone'),
            Patch(facecolor='red', alpha=0.2, label='Supply Zone'),
            Patch(facecolor='green', alpha=0.1, label='Uptrend'),
            Patch(facecolor='red', alpha=0.1, label='Downtrend'),
            Patch(facecolor='gray', alpha=0.1, label='Undefined Trend')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        plt.show()