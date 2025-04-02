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
    CandlestickVisualizer.plot_all_patterns(data, window=100)


if __name__ == "__main__":
    main()
