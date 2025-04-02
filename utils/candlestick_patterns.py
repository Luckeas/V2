import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional, Tuple


class CandlestickPatternDetector:
    """
    Utility class for detecting candlestick patterns in price data.
    
    This class provides methods to detect various candlestick patterns
    which can be used by trading strategies.
    """
    
    @staticmethod
    def detect_pattern(data: pd.DataFrame, pattern_name: str, **kwargs) -> np.ndarray:
        """
        Detect a specific candlestick pattern.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            pattern_name (str): Name of the pattern to detect.
            **kwargs: Additional parameters specific to the pattern.
            
        Returns:
            np.ndarray: Array of pattern signals (1 for bullish, -1 for bearish, 0 for no pattern).
        """
        # Dictionary mapping pattern names to detection methods
        pattern_methods = {
            'doji': CandlestickPatternDetector.detect_doji,
            'hammer': CandlestickPatternDetector.detect_hammer,
            'shooting_star': CandlestickPatternDetector.detect_shooting_star,
            'engulfing': CandlestickPatternDetector.detect_engulfing,
            'morning_star': CandlestickPatternDetector.detect_morning_star,
            'evening_star': CandlestickPatternDetector.detect_evening_star,
            'harami': CandlestickPatternDetector.detect_harami,
            'piercing_line': CandlestickPatternDetector.detect_piercing_line,
            'dark_cloud_cover': CandlestickPatternDetector.detect_dark_cloud_cover,
            'three_white_soldiers': CandlestickPatternDetector.detect_three_white_soldiers,
            'three_black_crows': CandlestickPatternDetector.detect_three_black_crows,
            'marubozu': CandlestickPatternDetector.detect_marubozu,
            'spinning_top': CandlestickPatternDetector.detect_spinning_top,
            'tweezer_top': CandlestickPatternDetector.detect_tweezer_top,
            'tweezer_bottom': CandlestickPatternDetector.detect_tweezer_bottom
        }
        
        if pattern_name not in pattern_methods:
            raise ValueError(f"Pattern '{pattern_name}' not supported. Available patterns: {list(pattern_methods.keys())}")
        
        # Call the appropriate pattern detection method
        return pattern_methods[pattern_name](data, **kwargs)
    
    @staticmethod
    def detect_doji(data: pd.DataFrame, tolerance: float = 0.1) -> np.ndarray:
        """
        Detect Doji candlestick pattern.
        
        A Doji has an open and close that are virtually equal.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            tolerance (float, optional): Maximum body/range ratio to consider as Doji. Defaults to 0.1.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Calculate body size as percentage of range
            body_size = abs(close_prices[i] - open_prices[i])
            range_size = high_prices[i] - low_prices[i]
            
            # Avoid division by zero
            if range_size == 0:
                continue
                
            body_percentage = body_size / range_size
            
            # Doji has a very small body (typically less than tolerance of range)
            if body_percentage < tolerance:
                # Determine if it's a bullish or bearish context
                if i > 1:
                    # Bullish if in a downtrend
                    if close_prices[i-1] < open_prices[i-1]:
                        signals[i] = 1
                    # Bearish if in an uptrend
                    elif close_prices[i-1] > open_prices[i-1]:
                        signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_marubozu(data: pd.DataFrame, shadow_threshold: float = 0.05) -> np.ndarray:
        """
        Detect Marubozu candlestick pattern.
        
        A Marubozu is a candlestick with no or very small shadows, indicating
        strong buying or selling pressure throughout the period.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            shadow_threshold (float, optional): Maximum shadow/range ratio. Defaults to 0.05.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(len(data)):
            # Calculate body size and full range
            body_size = abs(close_prices[i] - open_prices[i])
            full_range = high_prices[i] - low_prices[i]
            
            # Avoid division by zero
            if full_range == 0 or body_size == 0:
                continue
            
            # Calculate shadows
            if close_prices[i] >= open_prices[i]:  # Bullish candle
                upper_shadow = high_prices[i] - close_prices[i]
                lower_shadow = open_prices[i] - low_prices[i]
            else:  # Bearish candle
                upper_shadow = high_prices[i] - open_prices[i]
                lower_shadow = close_prices[i] - low_prices[i]
            
            upper_shadow_ratio = upper_shadow / full_range
            lower_shadow_ratio = lower_shadow / full_range
            
            # Check for Marubozu - very small or no shadows
            if upper_shadow_ratio <= shadow_threshold and lower_shadow_ratio <= shadow_threshold:
                # Bullish Marubozu
                if close_prices[i] > open_prices[i]:
                    signals[i] = 1
                # Bearish Marubozu
                else:
                    signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_spinning_top(data: pd.DataFrame, body_threshold: float = 0.3, 
                           shadow_threshold: float = 0.3) -> np.ndarray:
        """
        Detect Spinning Top candlestick pattern.
        
        A spinning top has a small body with upper and lower shadows that are
        roughly equal in length, indicating indecision in the market.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            body_threshold (float, optional): Maximum body/range ratio. Defaults to 0.3.
            shadow_threshold (float, optional): Maximum shadow difference ratio. Defaults to 0.3.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(len(data)):
            # Calculate body size and full range
            body_size = abs(close_prices[i] - open_prices[i])
            full_range = high_prices[i] - low_prices[i]
            
            # Avoid division by zero
            if full_range == 0:
                continue
            
            body_ratio = body_size / full_range
            
            # Calculate shadows
            if close_prices[i] >= open_prices[i]:  # Bullish candle
                upper_shadow = high_prices[i] - close_prices[i]
                lower_shadow = open_prices[i] - low_prices[i]
            else:  # Bearish candle
                upper_shadow = high_prices[i] - open_prices[i]
                lower_shadow = close_prices[i] - low_prices[i]
            
            # Calculate shadow difference ratio
            total_shadow = upper_shadow + lower_shadow
            if total_shadow == 0:
                continue
                
            shadow_diff_ratio = abs(upper_shadow - lower_shadow) / total_shadow
            
            # Spinning Top: small body, roughly equal shadows
            if body_ratio <= body_threshold and shadow_diff_ratio <= shadow_threshold:
                # In an uptrend, it's a potential bearish signal
                if i > 0 and close_prices[i-1] > open_prices[i-1]:
                    signals[i] = -1
                # In a downtrend, it's a potential bullish signal
                elif i > 0 and close_prices[i-1] < open_prices[i-1]:
                    signals[i] = 1
        
        return signals
    
    @staticmethod
    def detect_tweezer_top(data: pd.DataFrame, price_threshold: float = 0.001) -> np.ndarray:
        """
        Detect Tweezer Top pattern (bearish reversal).
        
        A tweezer top consists of two candlesticks with matching highs,
        where the first is bullish and the second is bearish.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            price_threshold (float, optional): Maximum difference in high prices. Defaults to 0.001.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        high_prices = data['high'].values
        open_prices = data['open'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Calculate high price percentage difference
            if high_prices[i-1] == 0:
                continue
                
            high_diff = abs(high_prices[i] - high_prices[i-1]) / high_prices[i-1]
            
            # Check for Tweezer Top pattern
            if (high_diff <= price_threshold and  # Matching highs
                close_prices[i-1] > open_prices[i-1] and  # First candle is bullish
                close_prices[i] < open_prices[i]):  # Second candle is bearish
                signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_tweezer_bottom(data: pd.DataFrame, price_threshold: float = 0.001) -> np.ndarray:
        """
        Detect Tweezer Bottom pattern (bullish reversal).
        
        A tweezer bottom consists of two candlesticks with matching lows,
        where the first is bearish and the second is bullish.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            price_threshold (float, optional): Maximum difference in low prices. Defaults to 0.001.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        low_prices = data['low'].values
        open_prices = data['open'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Calculate low price percentage difference
            if low_prices[i-1] == 0:
                continue
                
            low_diff = abs(low_prices[i] - low_prices[i-1]) / low_prices[i-1]
            
            # Check for Tweezer Bottom pattern
            if (low_diff <= price_threshold and  # Matching lows
                close_prices[i-1] < open_prices[i-1] and  # First candle is bearish
                close_prices[i] > open_prices[i]):  # Second candle is bullish
                signals[i] = 1
        
        return signals
    
    @staticmethod
    def detect_hammer(data: pd.DataFrame, body_threshold: float = 0.3, 
                     upper_shadow_threshold: float = 0.1, 
                     lower_shadow_threshold: float = 0.6) -> np.ndarray:
        """
        Detect Hammer pattern (bullish reversal).
        
        A hammer has a small body, little or no upper shadow, and a long lower shadow.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            body_threshold (float, optional): Maximum body/range ratio. Defaults to 0.3.
            upper_shadow_threshold (float, optional): Maximum upper shadow/range ratio. Defaults to 0.1.
            lower_shadow_threshold (float, optional): Minimum lower shadow/range ratio. Defaults to 0.6.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            body_size = abs(close_prices[i] - open_prices[i])
            total_range = high_prices[i] - low_prices[i]
            
            # Avoid division by zero
            if total_range == 0:
                continue
                
            body_percentage = body_size / total_range
            
            # Calculate shadows
            if close_prices[i] >= open_prices[i]:  # Bullish candle
                upper_shadow = high_prices[i] - close_prices[i]
                lower_shadow = open_prices[i] - low_prices[i]
            else:  # Bearish candle
                upper_shadow = high_prices[i] - open_prices[i]
                lower_shadow = close_prices[i] - low_prices[i]
            
            upper_shadow_percentage = upper_shadow / total_range
            lower_shadow_percentage = lower_shadow / total_range
            
            # Hammer criteria: small body, small upper shadow, long lower shadow
            if (body_percentage < body_threshold and 
                upper_shadow_percentage < upper_shadow_threshold and 
                lower_shadow_percentage > lower_shadow_threshold):
                
                # Hammer is only bullish if in a downtrend
                if i > 1 and (data['close'].iloc[i-1] < data['open'].iloc[i-1] or 
                              (i > 2 and data['close'].iloc[i-2] < data['close'].iloc[i-1])):
                    signals[i] = 1
        
        return signals
    
    @staticmethod
    def detect_shooting_star(data: pd.DataFrame, body_threshold: float = 0.3, 
                            lower_shadow_threshold: float = 0.1, 
                            upper_shadow_threshold: float = 0.6) -> np.ndarray:
        """
        Detect Shooting Star pattern (bearish reversal).
        
        A shooting star has a small body, little or no lower shadow, and a long upper shadow.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            body_threshold (float, optional): Maximum body/range ratio. Defaults to 0.3.
            lower_shadow_threshold (float, optional): Maximum lower shadow/range ratio. Defaults to 0.1.
            upper_shadow_threshold (float, optional): Minimum upper shadow/range ratio. Defaults to 0.6.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            body_size = abs(close_prices[i] - open_prices[i])
            total_range = high_prices[i] - low_prices[i]
            
            # Avoid division by zero
            if total_range == 0:
                continue
                
            body_percentage = body_size / total_range
            
            # Calculate shadows
            if close_prices[i] >= open_prices[i]:  # Bullish candle
                upper_shadow = high_prices[i] - close_prices[i]
                lower_shadow = open_prices[i] - low_prices[i]
            else:  # Bearish candle
                upper_shadow = high_prices[i] - open_prices[i]
                lower_shadow = close_prices[i] - low_prices[i]
            
            upper_shadow_percentage = upper_shadow / total_range
            lower_shadow_percentage = lower_shadow / total_range
            
            # Shooting Star criteria: small body, small lower shadow, long upper shadow
            if (body_percentage < body_threshold and 
                lower_shadow_percentage < lower_shadow_threshold and 
                upper_shadow_percentage > upper_shadow_threshold):
                
                # Shooting Star is only bearish if in an uptrend
                if i > 1 and (data['close'].iloc[i-1] > data['open'].iloc[i-1] or 
                              (i > 2 and data['close'].iloc[i-2] < data['close'].iloc[i-1])):
                    signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_engulfing(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Bullish and Bearish Engulfing patterns.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Current bar data
            current_open = open_prices[i]
            current_close = close_prices[i]
            current_body_size = abs(current_close - current_open)
            
            # Previous bar data
            prev_open = open_prices[i-1]
            prev_close = close_prices[i-1]
            prev_body_size = abs(prev_close - prev_open)
            
            # Check for bullish engulfing (current bar is bullish and engulfs previous bearish bar)
            if (current_close > current_open and  # Current bar is bullish
                prev_close < prev_open and        # Previous bar is bearish
                current_open < prev_close and     # Current open is below previous close
                current_close > prev_open and     # Current close is above previous open
                current_body_size > prev_body_size):  # Current body is larger
                signals[i] = 1
            
            # Check for bearish engulfing (current bar is bearish and engulfs previous bullish bar)
            elif (current_close < current_open and  # Current bar is bearish
                  prev_close > prev_open and        # Previous bar is bullish
                  current_open > prev_close and     # Current open is above previous close
                  current_close < prev_open and     # Current close is below previous open
                  current_body_size > prev_body_size):  # Current body is larger
                signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_morning_star(data: pd.DataFrame, doji_body_threshold: float = 0.3, 
                           price_threshold: float = 0.5) -> np.ndarray:
        """
        Detect Morning Star pattern (bullish reversal).
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            doji_body_threshold (float, optional): Maximum body/range ratio for middle candle. Defaults to 0.3.
            price_threshold (float, optional): Minimum 3rd candle close position relative to 1st candle. Defaults to 0.5.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(2, len(data)):
            # First candle (bearish)
            first_open = open_prices[i-2]
            first_close = close_prices[i-2]
            first_body_size = first_open - first_close  # Should be positive for bearish
            
            # Second candle (small body)
            second_open = open_prices[i-1]
            second_close = close_prices[i-1]
            second_body_size = abs(second_close - second_open)
            second_range = high_prices[i-1] - low_prices[i-1]
            
            # Third candle (bullish)
            third_open = open_prices[i]
            third_close = close_prices[i]
            third_body_size = third_close - third_open  # Should be positive for bullish
            
            # Avoid division by zero
            if second_range == 0:
                continue
                
            second_body_percentage = second_body_size / second_range
            
            # Check conditions for Morning Star
            if (first_body_size > 0 and  # First candle is bearish
                second_body_percentage < doji_body_threshold and  # Second candle has small body
                third_body_size > 0 and  # Third candle is bullish
                third_close > (first_open + first_close) * price_threshold):  # Third close is above threshold of first candle
                signals[i] = 1
        
        return signals

    @staticmethod
    def detect_evening_star(data: pd.DataFrame, star_body_threshold: float = 0.2,
                            price_threshold: float = 0.6, gap_threshold: float = 0.001) -> np.ndarray:
        """
        Detect Evening Star pattern (bearish reversal).

        An Evening Star consists of:
        1. A large bullish candle
        2. A small-bodied candle (the star) that gaps up from the first candle
        3. A bearish candle that gaps down from the star and closes deep into the first candle's body

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            star_body_threshold (float, optional): Maximum body/range ratio for star candle. Defaults to 0.2.
            price_threshold (float, optional): Minimum penetration into first candle body. Defaults to 0.6.
            gap_threshold (float, optional): Minimum price gap required. Defaults to 0.001.

        Returns:
            np.ndarray: Array of pattern signals (-1 for bearish evening star).
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        for i in range(2, len(data)):
            # First candle (bullish)
            first_open = open_prices[i - 2]
            first_close = close_prices[i - 2]
            first_high = high_prices[i - 2]
            first_body_size = first_close - first_open  # Should be positive for bullish

            # Second candle (star)
            second_open = open_prices[i - 1]
            second_close = close_prices[i - 1]
            second_high = high_prices[i - 1]
            second_low = low_prices[i - 1]
            second_body_size = abs(second_close - second_open)
            second_range = high_prices[i - 1] - low_prices[i - 1]

            # Third candle (bearish)
            third_open = open_prices[i]
            third_close = close_prices[i]
            third_high = high_prices[i]
            third_body_size = third_open - third_close  # Should be positive for bearish

            # Avoid division by zero
            if second_range == 0 or first_body_size <= 0 or third_body_size <= 0:
                continue

            # Calculate body percentage for the star candle
            second_body_percentage = second_body_size / second_range

            # Calculate gap up from first candle to star
            gap_up = second_low > first_close

            # Calculate gap down from star to third candle
            gap_down = third_high < second_low

            # Calculate penetration into first candle's body
            first_body_midpoint = first_open + (first_body_size / 2)
            penetration_percentage = (first_close - third_close) / first_body_size if first_body_size > 0 else 0

            # Check conditions for Evening Star
            if (first_body_size > 0 and  # First candle is bullish
                    second_body_percentage < star_body_threshold and  # Second candle has small body (star)
                    third_body_size > 0 and  # Third candle is bearish
                    gap_up and  # Star gaps up from first candle
                    penetration_percentage > price_threshold):  # Third candle closes deep into first candle
                signals[i] = -1

        return signals
    
    @staticmethod
    def detect_harami(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Bullish and Bearish Harami patterns.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Current bar data
            current_open = open_prices[i]
            current_close = close_prices[i]
            current_body_size = abs(current_close - current_open)
            
            # Previous bar data
            prev_open = open_prices[i-1]
            prev_close = close_prices[i-1]
            prev_body_size = abs(prev_close - prev_open)
            
            # Check for bullish harami (current bar is bullish and contained within previous bearish bar)
            if (current_close > current_open and  # Current bar is bullish
                prev_close < prev_open and        # Previous bar is bearish
                current_open > prev_close and     # Current open is higher than previous close
                current_close < prev_open and     # Current close is lower than previous open
                current_body_size < prev_body_size):  # Current body is smaller
                signals[i] = 1
            
            # Check for bearish harami (current bar is bearish and contained within previous bullish bar)
            elif (current_close < current_open and  # Current bar is bearish
                  prev_close > prev_open and        # Previous bar is bullish
                  current_open < prev_close and     # Current open is lower than previous close
                  current_close > prev_open and     # Current close is higher than previous open
                  current_body_size < prev_body_size):  # Current body is smaller
                signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_piercing_line(data: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Detect Piercing Line pattern (bullish reversal).
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            threshold (float, optional): Minimum close percentage into previous candle. Defaults to 0.5.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Previous bar (bearish)
            prev_open = open_prices[i-1]
            prev_close = close_prices[i-1]
            prev_midpoint = (prev_open + prev_close) / 2
            
            # Current bar (bullish)
            current_open = open_prices[i]
            current_close = close_prices[i]
            
            # Check for piercing line
            if (prev_close < prev_open and  # Previous bar is bearish
                current_close > current_open and  # Current bar is bullish
                current_open < prev_close and  # Gap down or open below previous close
                current_close > prev_midpoint and  # Close above 50% of previous body
                current_close < prev_open):  # But doesn't close above previous open
                signals[i] = 1
        
        return signals
    
    @staticmethod
    def detect_dark_cloud_cover(data: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Detect Dark Cloud Cover pattern (bearish reversal).
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            threshold (float, optional): Minimum close percentage into previous candle. Defaults to 0.5.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Previous bar (bullish)
            prev_open = open_prices[i-1]
            prev_close = close_prices[i-1]
            prev_midpoint = (prev_open + prev_close) / 2
            
            # Current bar (bearish)
            current_open = open_prices[i]
            current_close = close_prices[i]
            
            # Check for dark cloud cover
            if (prev_close > prev_open and  # Previous bar is bullish
                current_close < current_open and  # Current bar is bearish
                current_open > prev_close and  # Gap up or open above previous close
                current_close < prev_midpoint and  # Close below 50% of previous body
                current_close > prev_open):  # But doesn't close below previous open
                signals[i] = -1
        
        return signals
    
    @staticmethod
    def detect_three_white_soldiers(data: pd.DataFrame, shadow_threshold: float = 0.2) -> np.ndarray:
        """
        Detect Three White Soldiers pattern (bullish continuation).
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            shadow_threshold (float, optional): Maximum shadow/body ratio. Defaults to 0.2.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(2, len(data)):
            # Check if we have three consecutive bullish candles
            bullish_candles = (
                close_prices[i] > open_prices[i] and
                close_prices[i-1] > open_prices[i-1] and
                close_prices[i-2] > open_prices[i-2]
            )
            
            # Check if each candle opens within the previous candle's body
            progressive_opens = (
                open_prices[i] > open_prices[i-1] and
                open_prices[i-1] > open_prices[i-2]
            )
            
            # Check if each candle closes higher than the previous
            progressive_closes = (
                close_prices[i] > close_prices[i-1] and
                close_prices[i-1] > close_prices[i-2]
            )
            
            # Check for small upper shadows (showing strength)
            small_upper_shadows = True
            for j in range(i-2, i+1):
                body_size = close_prices[j] - open_prices[j]
                upper_shadow = high_prices[j] - close_prices[j]
                
                if body_size == 0 or upper_shadow / body_size > shadow_threshold:
                    small_upper_shadows = False
                    break
            
            if bullish_candles and progressive_opens and progressive_closes and small_upper_shadows:
                signals[i] = 1
        
        return signals
    
    @staticmethod
    def detect_three_black_crows(data: pd.DataFrame, shadow_threshold: float = 0.2) -> np.ndarray:
        """
        Detect Three Black Crows pattern (bearish continuation).
        
        A bearish reversal pattern consisting of three consecutive long-bodied 
        bearish candles that open within the real body of the previous candle
        and close lower than the previous candle.
        
        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            shadow_threshold (float, optional): Maximum shadow/body ratio. Defaults to 0.2.
            
        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        signals = np.zeros(len(data))
        
        for i in range(2, len(data)):
            # Check if we have three consecutive bearish candles
            bearish_candles = (
                close_prices[i] < open_prices[i] and
                close_prices[i-1] < open_prices[i-1] and
                close_prices[i-2] < open_prices[i-2]
            )
            
            # Check if each candle opens within the previous candle's body
            progressive_opens = (
                open_prices[i] < open_prices[i-1] and
                open_prices[i-1] < open_prices[i-2]
            )
            
            # Check if each candle closes lower than the previous
            progressive_closes = (
                close_prices[i] < close_prices[i-1] and
                close_prices[i-1] < close_prices[i-2]
            )
            
            # Check for small lower shadows (showing strength)
            small_lower_shadows = True
            for j in range(i-2, i+1):
                body_size = open_prices[j] - close_prices[j]
                lower_shadow = close_prices[j] - low_prices[j]
                
                if body_size == 0 or lower_shadow / body_size > shadow_threshold:
                    small_lower_shadows = False
                    break
            
            if bearish_candles and progressive_opens and progressive_closes and small_lower_shadows:
                signals[i] = -1
        
        return signals
        