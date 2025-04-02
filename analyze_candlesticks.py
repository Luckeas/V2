#!/usr/bin/env python3
"""
Utility script for analyzing candlestick patterns in market data.

This script allows you to:
1. Visualize specific candlestick patterns
2. Analyze the statistical performance of patterns
3. Scan for all supported candlestick patterns in one go
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader
from utils.candlestick_patterns import CandlestickPatternDetector
from utils.candlestick_visualizer import CandlestickVisualizer


def main():
    """Main function to run candlestick pattern analysis."""
    # Load your market data
    data_loader = DataLoader()

    # Replace this path with the path to your data file
    # You can use either load_csv or load_mes_futures based on your data format
    data = data_loader.load_mes_futures('/Users/martinshih/Downloads/Systematic/Candlestick_Data/MES_Data/U19_H25.csv')

    print(f"Loaded data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Option 1: Visualize a specific pattern
    pattern_type = 'engulfing'
    print(f"\nAnalyzing {pattern_type} patterns...")

    # Detect the pattern
    signals = CandlestickPatternDetector.detect_pattern(data, pattern_type)

    # Count the patterns found
    bullish_count = np.sum(signals > 0)
    bearish_count = np.sum(signals < 0)
    print(f"Found {bullish_count} bullish and {bearish_count} bearish {pattern_type} patterns")

    if bullish_count + bearish_count > 0:
        # Plot candlesticks with signals
        CandlestickVisualizer.plot_candlesticks(
            data,
            signals,
            window=50,  # Show 50 bars
            title=f"{pattern_type.title()} Pattern Detection"
        )

        # Analyze pattern performance
        CandlestickVisualizer.plot_pattern_statistics(
            data,
            signals,
            lookahead=5,  # Analyze return 5 bars after pattern
            title=f"{pattern_type.title()} Pattern Performance"
        )

    # Option 2: Scan for all patterns
    # Uncomment the line below to scan for all patterns
    # This will scan for all patterns and display visualizations for each
    print("\nScanning for all candlestick patterns...")
    scan_all_patterns(data)


def scan_all_patterns(data: pd.DataFrame, window: int = 100):
    """
    Scan for all available candlestick patterns and display results.

    Args:
        data (pd.DataFrame): Market data with OHLC prices.
        window (int, optional): Number of bars to show in visualizations. Defaults to 100.
    """
    # List of all patterns to detect
    patterns = [
        # Original patterns
        'doji', 'hammer', 'shooting_star', 'engulfing',
        'morning_star', 'evening_star', 'harami',
        'marubozu', 'spinning_top', 'tweezer_top', 'tweezer_bottom',
        'piercing_line', 'dark_cloud_cover', 'three_white_soldiers',
        'three_black_crows',

        # First set of patterns
        'mat_hold', 'deliberation', 'concealing_baby_swallow',
        'rising_three_methods', 'separating_lines',
        'falling_three_methods', 'doji_star',
        'last_engulfing_top', 'two_black_gapping',
        'side_by_side_white_lines',

        # Second set of patterns
        'three_stars_in_the_south', 'three_line_strike',
        'identical_three_crows', 'morning_doji_star',
        'three_outside_up',

        # Third set of patterns
        'three_line_strike_bearish', 'three_line_strike_bullish',
        'upside_tasuki_gap', 'hammer_inverted', 'matching_low',
        'abandoned_baby', 'breakaway_bearish'
    ]

    # Dictionary to store pattern statistics
    pattern_stats = {}

    for pattern in patterns:
        try:
            # Detect pattern
            signals = CandlestickPatternDetector.detect_pattern(data, pattern)

            # Count patterns
            bullish_count = np.sum(signals > 0)
            bearish_count = np.sum(signals < 0)
            total_count = bullish_count + bearish_count

            pattern_stats[pattern] = {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'total': total_count
            }

            # Print results
            print(f"{pattern.replace('_', ' ').title()}: {bullish_count} bullish, {bearish_count} bearish")

            # Visualize if patterns found
            if total_count > 0:
                # Plot candlesticks with signals
                CandlestickVisualizer.plot_candlesticks(
                    data, signals, window,
                    title=f"{pattern.replace('_', ' ').title()} Pattern Detection"
                )

                # Plot pattern statistics
                CandlestickVisualizer.plot_pattern_statistics(
                    data, signals, lookahead=5,
                    title=f"{pattern.replace('_', ' ').title()} Pattern Performance"
                )
        except Exception as e:
            print(f"Error detecting {pattern}: {e}")

    # Print summary
    print("\nPattern Detection Summary:")
    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    for pattern, stats in sorted_patterns:
        if stats['total'] > 0:
            print(f"{pattern.replace('_', ' ').title()}: {stats['total']} total patterns "
                  f"({stats['bullish']} bullish, {stats['bearish']} bearish)")


if __name__ == "__main__":
    main()