#!/usr/bin/env python3
"""
Script to test different exit strategies for the Morning Reversal strategy.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from utils.data_loader import DataLoader
from utils.config import Config
from engine.strategy_factory import StrategyFactory
from utils.exit_strategy_analyzer import ExitStrategyAnalyzer
from utils.plot import PerformancePlotter


def main():
    """Main function to test different exit strategies."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test different exit strategies for Morning Reversal')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--account', type=float, default=100000.0, help='Starting account value')
    parser.add_argument('--start_date', type=str, help='Start date for testing (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for testing (YYYY-MM-DD)')
    parser.add_argument('--mes_data', action='store_true', help='Indicate if the data is MES futures format')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_loader = DataLoader()
    
    if args.mes_data:
        data = data_loader.load_mes_futures(args.data)
        print(f"Loaded MES futures data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    else:
        data = data_loader.load_csv(args.data)
        print(f"Loaded data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")

    # Filter data by date range if provided
    if args.start_date:
        data = data[data.index >= args.start_date]
    if args.end_date:
        data = data[data.index <= args.end_date]
    
    print(f"Using date range: {data.index[0]} to {data.index[-1]} ({len(data)} bars)")

    # Load configuration
    config = Config.load_yaml(args.config)
    strategy_config = Config.get_strategy_config(config, 'morning_reversal')
    default_params = strategy_config.get('default_params', {})
    print(f"Loaded configuration with parameters: {default_params}")

    # Create Morning Reversal strategy
    strategy = StrategyFactory.create_strategy('morning_reversal', default_params)
    print(f"Created strategy: {strategy.name}")

    # Generate signals
    signals = strategy.generate_signals(data)
    signal_count = np.sum(signals != 0)
    print(f"Generated {signal_count} trading signals")

    # Create exit strategy analyzer
    analyzer = ExitStrategyAnalyzer(strategy, data, args.account)
    
    # Analyze different exit strategies
    print("\nAnalyzing exit strategies...")
    results = analyzer.analyze_exit_strategies(signals)
    
    # Plot comparison
    analyzer.plot_comparison()
    
    # Plot detailed equity curves using performance plotter
    plotter = PerformancePlotter()
    
    for name, result in results.items():
        print(f"\nDetailed analysis for {name.replace('_', ' ').title()} strategy:")
        returns_array = np.array(result['returns'])
        
        # Plot equity curve
        if isinstance(data.index, pd.DatetimeIndex):
            plotter.plot_equity_curve_with_time(
                returns_array, 
                data.index[:len(returns_array)], 
                title=f"Equity Curve - {name.replace('_', ' ').title()} Exit Strategy"
            )
        else:
            plotter.plot_equity_curve(
                returns_array, 
                title=f"Equity Curve - {name.replace('_', ' ').title()} Exit Strategy"
            )
        
        # Plot drawdown
        if isinstance(data.index, pd.DatetimeIndex):
            plotter.plot_drawdown_with_time(
                returns_array, 
                data.index[:len(returns_array)], 
                title=f"Drawdown - {name.replace('_', ' ').title()} Exit Strategy"
            )
        else:
