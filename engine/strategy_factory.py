from typing import Dict, Any, Optional

# Import all strategies
from strategies.base import Strategy
from strategies.donchian_breakout import DonchianBreakout
from strategies.moving_average_crossover import MovingAverageCrossover

# Dictionary mapping strategy names to their classes
STRATEGY_REGISTRY = {
    'donchian_breakout': DonchianBreakout,
    'moving_average_crossover': MovingAverageCrossover,
}


class StrategyFactory:
    """
    Factory class for creating strategy instances.
    """
    
    @staticmethod
    def create_strategy(strategy_name: str, params: Optional[Dict[str, Any]] = None) -> Strategy:
        """
        Create a strategy instance based on name and parameters.
        
        Args:
            strategy_name (str): Name of the strategy.
            params (Optional[Dict[str, Any]], optional): Strategy parameters. Defaults to None.
            
        Returns:
            Strategy: Strategy instance.
        """
        # Convert to lowercase and replace spaces with underscores
        strategy_key = strategy_name.lower().replace(' ', '_')
        
        if strategy_key not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry. "
                            f"Available strategies: {list(STRATEGY_REGISTRY.keys())}")
        
        # Get strategy class
        strategy_class = STRATEGY_REGISTRY[strategy_key]
        
        # Create instance with parameters
        if params is not None:
            return strategy_class(**params)
        else:
            return strategy_class()
    
    @staticmethod
    def register_strategy(strategy_name: str, strategy_class):
        """
        Register a new strategy in the registry.
        
        Args:
            strategy_name (str): Name of the strategy.
            strategy_class: Strategy class.
        """
        # Convert to lowercase and replace spaces with underscores
        strategy_key = strategy_name.lower().replace(' ', '_')
        
        STRATEGY_REGISTRY[strategy_key] = strategy_class
        
    @staticmethod
    def get_available_strategies() -> list:
        """
        Get list of available strategies.
        
        Returns:
            list: List of strategy names.
        """
        return list(STRATEGY_REGISTRY.keys())
