import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union, Callable


class ExitStrategyAnalyzer:
    """
    Utility class for analyzing different exit strategies.
    
    This class helps compare different exit strategies for trading systems,
    including nearest level, furthest level, and partial exit approaches.
    
    Exit strategies implemented:
    - Nearest Level: Exits at the closest key level (default strategy)
    - Furthest Level: Exits at the furthest key level for more ambitious targets
    - Partial Exit: Takes partial profits at nearest level and remainder at furthest level
    - Trailing Stop: Uses trailing stop that activates after reaching the nearest level
    """
    
    def __init__(self, strategy, data: pd.DataFrame, account_value: float = 100000.0):
        """
        Initialize the analyzer.
        
        Args:
            strategy: Trading strategy instance.
            data (pd.DataFrame): Market data with OHLC prices.
            account_value (float, optional): Starting account value. Defaults to 100000.0.
        """
        self.strategy = strategy
        self.data = data
        self.account_value = account_value
        self.results = {}
        
    def analyze_exit_strategies(self, signals: np.ndarray = None):
        """
        Analyze and compare different exit strategies using the given signals.
        
        Args:
            signals (np.ndarray, optional): Array of position signals. 
                                         If None, generates signals using the strategy.
        
        Returns:
            Dict: Dictionary of results for different exit strategies.
        """
        # Generate signals if not provided
        if signals is None:
            signals = self.strategy.generate_signals(self.data)
        
        # Compute returns for each exit strategy
        nearest_returns = self.compute_returns_nearest_level(signals)
        furthest_returns = self.compute_returns_furthest_level(signals)
        partial_returns = self.compute_returns_partial_exit(signals)
        trailing_returns = self.compute_returns_trailing_stop(signals)
        
        # Calculate metrics for each strategy
        def calculate_metrics(returns):
            # Start with initial account value
            equity = self.account_value
            equity_curve = [equity]
            
            # Calculate equity curve
            for r in returns:
                if r != 0:
                    equity *= (1 + r)
                equity_curve.append(equity)
            
            # Calculate drawdown
            peaks = np.maximum.accumulate(equity_curve)
            drawdown = (peaks - equity_curve) / peaks
            
            return {
                'total_return': (equity - self.account_value) / self.account_value,
                'annualized_return': (equity / self.account_value) ** (252 / len(returns)) - 1,
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                'max_drawdown': np.max(drawdown),
                'win_rate': np.sum(returns > 0) / np.sum(returns != 0) if np.sum(returns != 0) > 0 else 0,
                'profit_factor': np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) 
                               if np.sum(returns < 0) < 0 else float('inf'),
                'equity_curve': equity_curve,
                'returns': returns
            }
        
        # Store results
        self.results = {
            'nearest_level': calculate_metrics(nearest_returns),
            'furthest_level': calculate_metrics(furthest_returns),
            'partial_exit': calculate_metrics(partial_returns),
            'trailing_stop': calculate_metrics(trailing_returns)
        }
        
        return self.results
    
    def plot_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot a comparison of exit strategies.
        
        Args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        """
        if not self.results:
            print("No results to plot. Run analyze_exit_strategies() first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot equity curves
        for name, result in self.results.items():
            axes[0].plot(result['equity_curve'], label=name.replace('_', ' ').title())
        
        axes[0].set_title('Equity Curve Comparison')
        axes[0].set_ylabel('Account Value ($)')
        axes[0].axhline(y=self.account_value, color='black', linestyle='--', alpha=0.3)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot drawdown
        for name, result in self.results.items():
            equity_curve = np.array(result['equity_curve'])
            peaks = np.maximum.accumulate(equity_curve)
            drawdown = (peaks - equity_curve) / peaks * 100  # Convert to percentage
            axes[1].plot(drawdown, label=name.replace('_', ' ').title())
        
        axes[1].set_title('Drawdown Comparison')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Bar Number')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics table
        print("\nExit Strategy Comparison:")
        print(f"{'Strategy':<15} {'Total Return':<15} {'Ann. Return':<15} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10} {'Profit Factor':<15}")
        print("-" * 90)
        
        for name, result in self.results.items():
            print(f"{name.replace('_', ' ').title():<15} "
                  f"{result['total_return']:.2%:<15} "
                  f"{result['annualized_return']:.2%:<15} "
                  f"{result['sharpe_ratio']:.2f}:<10} "
                  f"{result['max_drawdown']:.2%:<10} "
                  f"{result['win_rate']:.2%:<10} "
                  f"{result['profit_factor']:.2f}:<15}")
        
        
    def compute_returns_nearest_level(self, signals: np.ndarray) -> np.ndarray:
        """
        Compute returns using the nearest key level exit strategy.
        
        Args:
            signals (np.ndarray): Array of position signals.
            
        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Default compute_returns implementation from the strategy
        return self.strategy.compute_returns(self.data, signals)
    
    def compute_returns_partial_exit(self, signals: np.ndarray) -> np.ndarray:
        """
        Compute returns using partial exit strategy.
        
        Takes 50% profit at nearest level and moves stop to breakeven,
        then takes remaining position profit at furthest level.
        
        Args:
            signals (np.ndarray): Array of position signals.
            
        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Check if we have the required indicators, if not, add them
        data = self.data
        if 'atr' not in data.columns:
            data = self.strategy._add_indicators(data)
        
        # Get key levels for targets
        key_levels = self.strategy._get_key_levels(data)
        
        # Initialize returns array
        returns = np.zeros(len(data))
        
        # Variables to track current position
        in_position = False
        entry_price = 0
        entry_index = 0
        position_type = 0  # 1 for long, -1 for short
        position_size = 1.0  # Full position
        stop_loss = 0
        first_target = 0
        second_target = 0
        partial_exit_done = False
        
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
                    position_size = 1.0  # Start with full position
                    entry_price = data['close'].iloc[i]
                    entry_index = i
                    partial_exit_done = False
                    
                    # Set stop loss
                    stop_loss = entry_price - (position_type * self.strategy.parameters['stop_atr_multiplier'] * data['atr'].iloc[i])
                    
                    # Set both targets - nearest and furthest
                    if position_type > 0:  # Long position
                        # Find nearest and furthest resistance levels
                        prev_day_high = key_levels.get('prev_day_high')
                        post_hours_high = key_levels.get('post_hours_high')
                        
                        if prev_day_high is not None and post_hours_high is not None:
                            # Set first target as the nearest level
                            if abs(prev_day_high - entry_price) <= abs(post_hours_high - entry_price):
                                first_target = prev_day_high
                                second_target = post_hours_high
                            else:
                                first_target = post_hours_high
                                second_target = prev_day_high
                        elif prev_day_high is not None:
                            first_target = prev_day_high
                            second_target = first_target * 1.01  # Default second target
                        elif post_hours_high is not None:
                            first_target = post_hours_high
                            second_target = first_target * 1.01  # Default second target
                        else:
                            # Default if no levels available
                            first_target = entry_price * 1.02
                            second_target = entry_price * 1.03
                    else:  # Short position
                        # Find nearest and furthest support levels
                        prev_day_low = key_levels.get('prev_day_low')
                        post_hours_low = key_levels.get('post_hours_low')
                        
                        if prev_day_low is not None and post_hours_low is not None:
                            # Set first target as the nearest level
                            if abs(prev_day_low - entry_price) <= abs(post_hours_low - entry_price):
                                first_target = prev_day_low
                                second_target = post_hours_low
                            else:
                                first_target = post_hours_low
                                second_target = prev_day_low
                        elif prev_day_low is not None:
                            first_target = prev_day_low
                            second_target = first_target * 0.99  # Default second target
                        elif post_hours_low is not None:
                            first_target = post_hours_low
                            second_target = first_target * 0.99  # Default second target
                        else:
                            # Default if no levels available
                            first_target = entry_price * 0.98
                            second_target = entry_price * 0.97
            else:
                # Check for exit conditions
                current_price = data['close'].iloc[i]
                exit_signal = False
                exit_reason = None
                exit_price = 0
                exit_size = 0
                
                # First check stop loss
                if (position_type > 0 and data['low'].iloc[i] <= stop_loss) or \
                   (position_type < 0 and data['high'].iloc[i] >= stop_loss):
                    exit_signal = True
                    exit_reason = "stop_loss"
                    exit_price = stop_loss
                    exit_size = position_size  # Exit entire remaining position
                
                # Check for first target (partial exit) if not yet done
                elif not partial_exit_done and ((position_type > 0 and data['high'].iloc[i] >= first_target) or \
                     (position_type < 0 and data['low'].iloc[i] <= first_target)):
                    exit_signal = True
                    exit_reason = "partial_exit"
                    exit_price = first_target
                    exit_size = 0.5  # Exit half position
                    
                    # Move stop loss to breakeven for remaining position
                    stop_loss = entry_price
                    partial_exit_done = True
                
                # Check for second target if partial exit was already done
                elif partial_exit_done and ((position_type > 0 and data['high'].iloc[i] >= second_target) or \
                     (position_type < 0 and data['low'].iloc[i] <= second_target)):
                    exit_signal = True
                    exit_reason = "final_exit"
                    exit_price = second_target
                    exit_size = position_size  # Exit remaining position
                
                # Execute exit if conditions met
                if exit_signal:
                    # Calculate log return for the partial or full exit
                    if position_type > 0:  # Long position
                        this_return = np.log(exit_price / entry_price) * exit_size
                    else:  # Short position
                        this_return = np.log(entry_price / exit_price) * exit_size
                    
                    returns[i] = this_return
                    
                    # Update position size
                    position_size -= exit_size
                    
                    # Reset position if completely exited
                    if position_size <= 0:
                        in_position = False
                        position_type = 0
                        entry_price = 0
                        stop_loss = 0
                        first_target = 0
                        second_target = 0
                        partial_exit_done = False
        
        return returns
    
    def compute_returns_trailing_stop(self, signals: np.ndarray) -> np.ndarray:
        """
        Compute returns using trailing stop exit strategy.
        
        Uses a trailing stop that activates after price reaches the nearest key level.
        
        Args:
            signals (np.ndarray): Array of position signals.
            
        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Check if we have the required indicators, if not, add them
        data = self.data
        if 'atr' not in data.columns:
            data = self.strategy._add_indicators(data)
        
        # Get key levels for targets
        key_levels = self.strategy._get_key_levels(data)
        
        # Initialize returns array
        returns = np.zeros(len(data))
        
        # Variables to track current position
        in_position = False
        entry_price = 0
        entry_index = 0
        position_type = 0  # 1 for long, -1 for short
        stop_loss = 0
        target = 0
        trailing_activated = False
        highest_price = 0
        lowest_price = 0
        trailing_pct = 0.005  # 0.5% trailing stop
        
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
                    trailing_activated = False
                    
                    # Initialize price tracking for trailing stop
                    highest_price = entry_price
                    lowest_price = entry_price
                    
                    # Set initial stop loss
                    stop_loss = entry_price - (position_type * self.strategy.parameters['stop_atr_multiplier'] * data['atr'].iloc[i])
                    
                    # Set target based on nearest key level
                    if position_type > 0:  # Long position
                        # Find the nearest resistance level
                        prev_day_high = key_levels.get('prev_day_high')
                        post_hours_high = key_levels.get('post_hours_high')
                        
                        if prev_day_high is not None and post_hours_high is not None:
                            # Use the closer of the two levels
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
                        # Find the nearest support level
                        prev_day_low = key_levels.get('prev_day_low')
                        post_hours_low = key_levels.get('post_hours_low')
                        
                        if prev_day_low is not None and post_hours_low is not None:
                            # Use the closer of the two levels
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
                # Update highest/lowest price for trailing stop
                if position_type > 0:  # Long position
                    highest_price = max(highest_price, data['high'].iloc[i])
                else:  # Short position
                    if lowest_price == 0:  # Initialize if not set
                        lowest_price = data['low'].iloc[i]
                    else:
                        lowest_price = min(lowest_price, data['low'].iloc[i])
                
                # Check if price has reached the target to activate trailing stop
                if not trailing_activated:
                    if (position_type > 0 and data['high'].iloc[i] >= target) or \
                       (position_type < 0 and data['low'].iloc[i] <= target):
                        trailing_activated = True
                
                # Update trailing stop if activated
                if trailing_activated:
                    if position_type > 0:  # Long position
                        new_stop = highest_price * (1 - trailing_pct)
                        if new_stop > stop_loss:
                            stop_loss = new_stop
                    else:  # Short position
                        new_stop = lowest_price * (1 + trailing_pct)
                        if stop_loss == 0 or new_stop < stop_loss:
                            stop_loss = new_stop
                
                # Check for exit conditions
                exit_signal = False
                exit_reason = None
                
                # Check stop loss
                if (position_type > 0 and data['low'].iloc[i] <= stop_loss) or \
                   (position_type < 0 and data['high'].iloc[i] >= stop_loss):
                    exit_signal = True
                    exit_reason = "stop_loss"
                    exit_price = stop_loss
                
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
                    trailing_activated = False
                    highest_price = 0
                    lowest_price = 0
        
        return returns
    
    def compute_returns_furthest_level(self, signals: np.ndarray) -> np.ndarray:
        """
        Compute returns using the furthest key level exit strategy.
        
        Args:
            signals (np.ndarray): Array of position signals.
            
        Returns:
            np.ndarray: Array of strategy returns.
        """
        # Check if we have the required indicators, if not, add them
        data = self.data
        if 'atr' not in data.columns:
            data = self.strategy._add_indicators(data)
        
        # Get key levels for targets
        key_levels = self.strategy._get_key_levels(data)
        
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
                    stop_loss = entry_price - (position_type * self.strategy.parameters['stop_atr_multiplier'] * data['atr'].iloc[i])
                    
                    # Set target based on furthest key level for more ambitious targets
                    if position_type > 0:  # Long position
                        # Use the furthest resistance level for long positions
                        prev_day_high = key_levels.get('prev_day_high')
                        post_hours_high = key_levels.get('post_hours_high')
                        
                        # Find the furthest target level
                        if prev_day_high is not None and post_hours_high is not None:
                            # Use the higher of the two levels
                            target = max(prev_day_high, post_hours_high)
                        elif prev_day_high is not None:
                            target = prev_day_high
                        elif post_hours_high is not None:
                            target = post_hours_high
                        else:
                            # Default if no levels available
                            target = entry_price * 1.03  # More ambitious than default
                    else:  # Short position
                        # Use the furthest support level for short positions
                        prev_day_low = key_levels.get('prev_day_low')
                        post_hours_low = key_levels.get('post_hours_low')
                        
                        # Find the furthest target level
                        if prev_day_low is not None and post_hours_low is not None:
                            # Use the lower of the two levels
                            target = min(prev_day_low, post_hours_low)
                        elif prev_day_low is not None:
                            target = prev_day_low
                        elif post_hours_low is not None:
                            target = post_hours_low
                        else:
                            # Default if no levels available
                            target = entry_price * 0.97  # More ambitious than default
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
