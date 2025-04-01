import yaml
import os
from typing import Dict, Any, List, Optional


class Config:
    """
    Utility class for loading and handling configuration.
    """
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            filepath (str): Path to the YAML file.
            
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def get_strategy_config(config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        """
        Get strategy-specific configuration.
        
        Args:
            config (Dict[str, Any]): Global configuration dictionary.
            strategy_name (str): Name of the strategy.
            
        Returns:
            Dict[str, Any]: Strategy-specific configuration.
        """
        if 'strategies' not in config:
            raise KeyError("No strategies section found in config.")
        
        strategies = config['strategies']
        
        if strategy_name not in strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found in config.")
        
        return strategies[strategy_name]
    
    @staticmethod
    def get_parameter_grid(strategy_config: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Get parameter grid from strategy configuration.
        
        Args:
            strategy_config (Dict[str, Any]): Strategy-specific configuration.
            
        Returns:
            Dict[str, List[Any]]: Parameter grid for optimization.
        """
        if 'parameter_grid' not in strategy_config:
            raise KeyError("No parameter_grid section found in strategy config.")
        
        parameter_grid = {}
        
        for param, param_config in strategy_config['parameter_grid'].items():
            if 'range' in param_config:
                # Parse range parameters
                start = param_config['range']['start']
                end = param_config['range']['end']
                step = param_config['range'].get('step', 1)
                
                # Generate parameter values
                param_values = list(range(start, end + 1, step))
            elif 'values' in param_config:
                # Use explicit values
                param_values = param_config['values']
            else:
                raise KeyError(f"Invalid parameter grid configuration for {param}.")
            
            parameter_grid[param] = param_values
        
        return parameter_grid
