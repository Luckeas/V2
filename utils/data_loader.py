import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict, Optional


class DataLoader:
    """
    Utility class for loading and preprocessing market data.
    """
    
    @staticmethod
    def load_csv(filepath: str, date_column: str = 'timestamp', 
                 datetime_format: Optional[str] = None) -> pd.DataFrame:
        """
        Load market data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            date_column (str, optional): Name of the date column. Defaults to 'timestamp'.
            datetime_format (Optional[str], optional): Format of the datetime. Defaults to None.
            
        Returns:
            pd.DataFrame: DataFrame with market data.
        """
        df = pd.read_csv(filepath)
        
        # Convert date column to datetime
        if datetime_format:
            df[date_column] = pd.to_datetime(df[date_column], format=datetime_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Set date column as index
        df.set_index(date_column, inplace=True)
        
        # Ensure columns are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have OHLC columns
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"Missing {col} column in the data.")
        
        return df
    
    @staticmethod
    def split_data(data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            data (pd.DataFrame): Market data.
            train_ratio (float, optional): Ratio of training data. Defaults to 0.7.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing data.
        """
        train_size = int(len(data) * train_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        return train_data, test_data
    
    @staticmethod
    def split_by_date(data: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by a specific date.
        
        Args:
            data (pd.DataFrame): Market data.
            split_date (str): Date to split on (e.g., '2020-01-01').
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Data before and after the split date.
        """
        before = data.loc[:split_date]
        after = data.loc[split_date:]
        
        return before, after
    
    @staticmethod
    def split_by_year(data: pd.DataFrame, years: List[int]) -> Dict[int, pd.DataFrame]:
        """
        Split data by years.
        
        Args:
            data (pd.DataFrame): Market data.
            years (List[int]): List of years to split on.
            
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping years to corresponding data.
        """
        result = {}
        
        for year in years:
            year_data = data[data.index.year == year]
            if not year_data.empty:
                result[year] = year_data
        
        return result
