# Save as test_permutation.py in your main project directory
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from engine.strategy_factory import StrategyFactory
from utils.permutation_test import insample_permutation_test
from utils.metrics import sharpe_ratio

# Load a small sample of data
data_loader = DataLoader()
data = data_loader.load_mes_futures('/Users/martinshih/Downloads/Systematic/Candlestick_Data/MES_Data/U19_H25.csv')
data = data.loc['2023-01-01':'2023-01-15']  # Just use 15 days

# Create strategy
strategy = StrategyFactory.create_strategy('enhanced_market_regime')

# Define minimal parameter grid
parameter_grid = {
    'rsi_oversold': [35],
    'rsi_overbought': [65],
    'volume_multiplier': [1.5],
    'max_bars_held': [2],
    'bb_window': [20],
    'stop_atr_multiplier': [1.0]
}

# Run direct permutation test
print("Starting direct permutation test...")
p_value, best_params, real_objective, perm_objectives = insample_permutation_test(
    strategy, data, sharpe_ratio, parameter_grid, n_permutations=5, show_plot=False
)
print(f"Test complete. P-value: {p_value}")
