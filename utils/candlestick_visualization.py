import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, Union, List, Optional, Tuple


class CandlestickVisualizer:
    """
    Utility class for visualizing candlestick patterns in price data.
    
    This class provides methods to visualize candlestick patterns
    identified by the CandlestickPatternDetector.
    """
    
    @staticmethod
    def plot_candlesticks(data: pd.DataFrame, 
                         signals: np.ndarray = None, 
                         window: int = 50,
                         start_index: int = None,
                         title: str = "Candlestick Chart",
                         figsize: Tuple[int, int] = (14, 7)) -> None:
        """
        Plot candlesticks with optional pattern signals.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray, optional): Pattern signals. Defaults to None.
            window (int, optional): Number of bars to show. Defaults to 50.
            start_index (int, optional): Starting index. Defaults to None.
            title (str, optional): Plot title. Defaults to "Candlestick Chart".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (14, 7).
        """
        # Convert to pandas DataFrame if numpy array
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
        
        # Use timestamp as index if not datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            dates = pd.date_range(start='2020-01-01', periods=len(data))
            data = data.set_index(dates)
        
        # Determine the window to plot
        if start_index is None:
            if signals is not None:
                # Find the first signal
                signal_indices = np.where(signals != 0)[0]
                if len(signal_indices) > 0:
                    start_index = max(0, signal_indices[0] - 10)
                else:
                    start_index = 0
            else:
                start_index = 0
        
        end_index = min(start_index + window, len(data))
        
        # Get the data slice to plot
        plot_data = data.iloc[start_index:end_index]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot candlesticks
        width = 0.8
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            # Calculate positions
            x = i
            
            # Plot candle body
            if row['close'] >= row['open']:
                # Bullish candle (green/white)
                color = 'green'
                body_bottom = row['open']
                body_height = row['close'] - row['open']
            else:
                # Bearish candle (red/black)
                color = 'red'
                body_bottom = row['close']
                body_height = row['open'] - row['close']
            
            # Draw body
            rect = Rectangle((x - width/2, body_bottom), width, body_height, 
                           facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            
            # Draw upper shadow
            ax.plot([x, x], [max(row['open'], row['close']), row['high']], 
                   color='black', linewidth=1)
            
            # Draw lower shadow
            ax.plot([x, x], [min(row['open'], row['close']), row['low']], 
                   color='black', linewidth=1)
        
        # Plot signals if provided
        if signals is not None:
            signal_slice = signals[start_index:end_index]
            
            for i, signal in enumerate(signal_slice):
                if signal > 0:  # Bullish signal
                    ax.annotate('↑', 
                              xy=(i, plot_data.iloc[i]['low']), 
                              xytext=(0, -20),
                              textcoords='offset points',
                              ha='center', 
                              va='top',
                              color='green',
                              fontsize=15,
                              arrowprops=dict(arrowstyle='->', color='green'))
                elif signal < 0:  # Bearish signal
                    ax.annotate('↓', 
                              xy=(i, plot_data.iloc[i]['high']), 
                              xytext=(0, 20),
                              textcoords='offset points',
                              ha='center', 
                              va='bottom',
                              color='red',
                              fontsize=15,
                              arrowprops=dict(arrowstyle='->', color='red'))
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Set x-axis ticks
        ax.set_xticks(range(0, len(plot_data), 5))
        ax.set_xticklabels([plot_data.index[i].strftime('%Y-%m-%d') for i in range(0, len(plot_data), 5)], 
                          rotation=45)
        
        # Set grid and tight layout
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pattern_statistics(data: pd.DataFrame, signals: np.ndarray, 
                               lookahead: int = 5, title: str = None) -> None:
        """
        Plot statistics about pattern performance.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            signals (np.ndarray): Pattern signals.
            lookahead (int, optional): Number of bars to look ahead. Defaults to 5.
            title (str, optional): Plot title. Defaults to None.
        """
        # Find signal indices
        bullish_indices = np.where(signals > 0)[0]
        bearish_indices = np.where(signals < 0)[0]
        
        # Calculate return statistics
        returns_after_bullish = []
        returns_after_bearish = []
        
        # For each bullish signal, calculate return after lookahead bars
        for idx in bullish_indices:
            if idx + lookahead < len(data):
                entry_price = data.iloc[idx]['close']
                exit_price = data.iloc[idx + lookahead]['close']
                returns_after_bullish.append((exit_price / entry_price) - 1)
        
        # For each bearish signal, calculate return after lookahead bars
        for idx in bearish_indices:
            if idx + lookahead < len(data):
                entry_price = data.iloc[idx]['close']
                exit_price = data.iloc[idx + lookahead]['close']
                # Invert the return for bearish signals
                returns_after_bearish.append(1 - (exit_price / entry_price))
        
        # Create figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot returns after bullish signals
        if returns_after_bullish:
            axes[0].hist(returns_after_bullish, bins=20, alpha=0.7, color='green')
            axes[0].axvline(x=0, color='black', linestyle='--')
            axes[0].axvline(x=np.mean(returns_after_bullish), color='red', linestyle='-')
            axes[0].set_title(f'Returns {lookahead} Bars After Bullish Signals (n={len(returns_after_bullish)})')
            axes[0].set_xlabel('Return')
            axes[0].set_ylabel('Frequency')
            axes[0].annotate(f'Mean: {np.mean(returns_after_bullish):.4f}\nMedian: {np.median(returns_after_bullish):.4f}\n'
                           f'Win Rate: {np.mean(np.array(returns_after_bullish) > 0):.2f}',
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           va='top', ha='left',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, 'No bullish signals found', ha='center', va='center')
        
        # Plot returns after bearish signals
        if returns_after_bearish:
            axes[1].hist(returns_after_bearish, bins=20, alpha=0.7, color='red')
            axes[1].axvline(x=0, color='black', linestyle='--')
            axes[1].axvline(x=np.mean(returns_after_bearish), color='green', linestyle='-')
            axes[1].set_title(f'Returns {lookahead} Bars After Bearish Signals (n={len(returns_after_bearish)})')
            axes[1].set_xlabel('Return')
            axes[1].set_ylabel('Frequency')
            axes[1].annotate(f'Mean: {np.mean(returns_after_bearish):.4f}\nMedian: {np.median(returns_after_bearish):.4f}\n'
                           f'Win Rate: {np.mean(np.array(returns_after_bearish) > 0):.2f}',
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           va='top', ha='left',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No bearish signals found', ha='center', va='center')
        
        # Set main title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_all_patterns(data: pd.DataFrame, window: int = 100, start_index: int = None) -> None:
        """
        Plot all common candlestick patterns for analysis.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            window (int, optional): Number of bars to show. Defaults to 100.
            start_index (int, optional): Starting index. Defaults to None.
        """
        from utils.candlestick_patterns import CandlestickPatternDetector
        
        # List of common patterns to detect
        patterns = [
            'doji', 'hammer', 'shooting_star', 'engulfing',
            'morning_star', 'evening_star', 'harami',
            'marubozu', 'spinning_top'
        ]
        
        for pattern in patterns:
            # Detect pattern
            signals = CandlestickPatternDetector.detect_pattern(data, pattern)
            
            # Check if any signals found
            if np.any(signals != 0):
                print(f"{pattern.replace('_', ' ').title()}: {np.sum(signals != 0)} patterns detected")
                
                # Plot candlesticks with signals
                CandlestickVisualizer.plot_candlesticks(
                    data, signals, window, start_index,
                    title=f"{pattern.replace('_', ' ').title()} Pattern Detection"
                )
                
                # Plot pattern statistics
                CandlestickVisualizer.plot_pattern_statistics(
                    data, signals, lookahead=5,
                    title=f"{pattern.replace('_', ' ').title()} Pattern Performance"
                )
            else:
                print(f"{pattern.replace('_', ' ').title()}: No patterns detected")
