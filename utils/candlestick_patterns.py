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
            'tweezer_bottom': CandlestickPatternDetector.detect_tweezer_bottom,
            # First set of patterns
            'mat_hold': CandlestickPatternDetector.detect_mat_hold,
            'deliberation': CandlestickPatternDetector.detect_deliberation,
            'concealing_baby_swallow': CandlestickPatternDetector.detect_concealing_baby_swallow,
            'rising_three_methods': CandlestickPatternDetector.detect_rising_three_methods,
            'separating_lines': CandlestickPatternDetector.detect_separating_lines,
            'falling_three_methods': CandlestickPatternDetector.detect_falling_three_methods,
            'doji_star': CandlestickPatternDetector.detect_doji_star,
            'last_engulfing_top': CandlestickPatternDetector.detect_last_engulfing_top,
            'two_black_gapping': CandlestickPatternDetector.detect_two_black_gapping,
            'side_by_side_white_lines': CandlestickPatternDetector.detect_side_by_side_white_lines,
            # Second set of patterns
            'three_stars_in_the_south': CandlestickPatternDetector.detect_three_stars_in_the_south,
            'three_line_strike': CandlestickPatternDetector.detect_three_line_strike,
            'identical_three_crows': CandlestickPatternDetector.detect_identical_three_crows,
            'morning_doji_star': CandlestickPatternDetector.detect_morning_doji_star,
            'three_outside_up': CandlestickPatternDetector.detect_three_outside_up,
            # Third set of patterns
            'three_line_strike_bearish': CandlestickPatternDetector.detect_three_line_strike_bearish,
            'three_line_strike_bullish': CandlestickPatternDetector.detect_three_line_strike_bullish,
            'upside_tasuki_gap': CandlestickPatternDetector.detect_upside_tasuki_gap,
            'hammer_inverted': CandlestickPatternDetector.detect_hammer_inverted,
            'matching_low': CandlestickPatternDetector.detect_matching_low,
            'abandoned_baby': CandlestickPatternDetector.detect_abandoned_baby,
            'breakaway_bearish': CandlestickPatternDetector.detect_breakaway_bearish
        }

        if pattern_name not in pattern_methods:
            raise ValueError(
                f"Pattern '{pattern_name}' not supported. Available patterns: {list(pattern_methods.keys())}")

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

    @staticmethod
    def detect_mat_hold(data: pd.DataFrame, gap_threshold: float = 0.003) -> np.ndarray:
        """
        Detect Mat Hold pattern (bullish continuation).

        A Mat Hold is a rare bullish continuation pattern consisting of:
        1. A strong bullish candle
        2. A gap up followed by 2-3 small bearish candles that stay above the first candle's open
        3. A final bullish candle that closes above the high of the first candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            gap_threshold (float, optional): Minimum size for the gap. Defaults to 0.003.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 5 candles to detect this pattern
        for i in range(4, len(data)):
            # First candle should be bullish and strong
            first_bullish = close_prices[i - 4] > open_prices[i - 4]
            first_range = high_prices[i - 4] - low_prices[i - 4]
            first_body = close_prices[i - 4] - open_prices[i - 4]

            # Check for strong first candle (body is at least 70% of range)
            if not (first_bullish and first_body >= 0.7 * first_range):
                continue

            # Check for gap up after first candle
            gap_up = low_prices[i - 3] > open_prices[i - 4]

            # Check for 2-3 small bearish candles that stay above first candle's open
            middle_bearish = (
                    close_prices[i - 3] < open_prices[i - 3] and
                    close_prices[i - 2] < open_prices[i - 2] and
                    close_prices[i - 1] < open_prices[i - 1]
            )

            above_first_open = (
                    low_prices[i - 3] > open_prices[i - 4] and
                    low_prices[i - 2] > open_prices[i - 4] and
                    low_prices[i - 1] > open_prices[i - 4]
            )

            # Check for small bodies in the middle candles
            small_bodies = (
                    abs(close_prices[i - 3] - open_prices[i - 3]) < 0.5 * first_body and
                    abs(close_prices[i - 2] - open_prices[i - 2]) < 0.5 * first_body and
                    abs(close_prices[i - 1] - open_prices[i - 1]) < 0.5 * first_body
            )

            # Final candle should be bullish and close above the high of first candle
            final_bullish = close_prices[i] > open_prices[i]
            closes_higher = close_prices[i] > high_prices[i - 4]

            if (first_bullish and gap_up and middle_bearish and above_first_open and
                    small_bodies and final_bullish and closes_higher):
                signals[i] = 1

        return signals

    @staticmethod
    def detect_deliberation(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Deliberation pattern (bearish reversal after uptrend).

        The Deliberation pattern consists of:
        1. Two strong bullish candles in an uptrend
        2. A third candle with a small body (often a spinning top or doji)

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles to detect this pattern
        for i in range(2, len(data)):
            # Check for uptrend and two bullish candles
            uptrend = (i > 3 and close_prices[i - 3] > close_prices[i - 4]) or True
            first_bullish = close_prices[i - 2] > open_prices[i - 2]
            second_bullish = close_prices[i - 1] > open_prices[i - 1]

            # First two candles should be strong
            first_body = close_prices[i - 2] - open_prices[i - 2]
            second_body = close_prices[i - 1] - open_prices[i - 1]
            first_range = high_prices[i - 2] - low_prices[i - 2]
            second_range = high_prices[i - 1] - low_prices[i - 1]

            strong_first_candles = (
                    first_body > 0.5 * first_range and
                    second_body > 0.5 * second_range
            )

            # Third candle has a small body
            third_body = abs(close_prices[i] - open_prices[i])
            third_range = high_prices[i] - low_prices[i]

            if third_range == 0:
                continue

            small_third_body = third_body < 0.3 * third_range

            # Check for gap up or higher open on each day
            advancing_opens = (
                    open_prices[i - 1] >= open_prices[i - 2] and
                    open_prices[i] >= open_prices[i - 1]
            )

            if (uptrend and first_bullish and second_bullish and strong_first_candles and
                    small_third_body and advancing_opens):
                signals[i] = -1  # This is a bearish reversal signal

        return signals

    @staticmethod
    def detect_concealing_baby_swallow(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Concealing Baby Swallow pattern (bullish reversal after downtrend).

        A complex pattern consisting of:
        1. Two bearish marubozu candles
        2. A third bearish candle that gaps down and has an upper shadow that penetrates the body of the 2nd candle
        3. A fourth bearish candle that completely engulfs the third candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 4 candles to detect this pattern
        for i in range(3, len(data)):
            # First two candles should be bearish marubozus
            first_bearish = close_prices[i - 3] < open_prices[i - 3]
            second_bearish = close_prices[i - 2] < open_prices[i - 2]

            # Check for small or no shadows on first two candles
            first_upper_shadow = high_prices[i - 3] - open_prices[i - 3]
            first_lower_shadow = close_prices[i - 3] - low_prices[i - 3]
            second_upper_shadow = high_prices[i - 2] - open_prices[i - 2]
            second_lower_shadow = close_prices[i - 2] - low_prices[i - 2]

            first_body = open_prices[i - 3] - close_prices[i - 3]
            second_body = open_prices[i - 2] - close_prices[i - 2]

            if first_body == 0 or second_body == 0:
                continue

            first_marubozu = (
                    first_bearish and
                    first_upper_shadow < 0.1 * first_body and
                    first_lower_shadow < 0.1 * first_body
            )

            second_marubozu = (
                    second_bearish and
                    second_upper_shadow < 0.1 * second_body and
                    second_lower_shadow < 0.1 * second_body
            )

            # Third candle should be bearish with gap down and upper shadow penetrating second candle
            third_bearish = close_prices[i - 1] < open_prices[i - 1]
            gap_down = open_prices[i - 1] < close_prices[i - 2]
            penetrates_second = high_prices[i - 1] > close_prices[i - 2]

            # Fourth candle should be bearish and engulf the third
            fourth_bearish = close_prices[i] < open_prices[i]
            engulfs_third = (
                    open_prices[i] >= open_prices[i - 1] and
                    close_prices[i] <= close_prices[i - 1] and
                    high_prices[i] >= high_prices[i - 1] and
                    low_prices[i] <= low_prices[i - 1]
            )

            if (first_marubozu and second_marubozu and third_bearish and gap_down and
                    penetrates_second and fourth_bearish and engulfs_third):
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_rising_three_methods(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Rising Three Methods pattern (bullish continuation).

        The pattern consists of:
        1. A strong bullish candle
        2. Three small bearish candles that stay within the range of the first candle
        3. A final bullish candle that closes above the close of the first candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 5 candles to detect this pattern
        for i in range(4, len(data)):
            # First candle should be a strong bullish candle
            first_bullish = close_prices[i - 4] > open_prices[i - 4]
            first_body = close_prices[i - 4] - open_prices[i - 4]
            first_range = high_prices[i - 4] - low_prices[i - 4]

            strong_first = first_bullish and first_body > 0.6 * first_range

            # Middle three candles should be bearish and small
            middle_bearish = (
                    close_prices[i - 3] < open_prices[i - 3] and
                    close_prices[i - 2] < open_prices[i - 2] and
                    close_prices[i - 1] < open_prices[i - 1]
            )

            # Middle candles should stay within the high-low range of the first candle
            stay_within_range = (
                    high_prices[i - 3] <= high_prices[i - 4] and
                    high_prices[i - 2] <= high_prices[i - 4] and
                    high_prices[i - 1] <= high_prices[i - 4] and
                    low_prices[i - 3] >= low_prices[i - 4] and
                    low_prices[i - 2] >= low_prices[i - 4] and
                    low_prices[i - 1] >= low_prices[i - 4]
            )

            # Small bodies in the middle
            middle_bodies = (
                    abs(close_prices[i - 3] - open_prices[i - 3]) < 0.5 * first_body and
                    abs(close_prices[i - 2] - open_prices[i - 2]) < 0.5 * first_body and
                    abs(close_prices[i - 1] - open_prices[i - 1]) < 0.5 * first_body
            )

            # Last candle should be bullish and close above the first candle's close
            last_bullish = close_prices[i] > open_prices[i]
            closes_higher = close_prices[i] >= close_prices[i - 4]

            if (strong_first and middle_bearish and stay_within_range and
                    middle_bodies and last_bullish and closes_higher):
                signals[i] = 1

        return signals

    @staticmethod
    def detect_separating_lines(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Separating Lines pattern (continuation pattern).

        Bullish Separating Lines:
        1. A bearish candle
        2. A bullish candle with the same open price but higher close

        Bearish Separating Lines:
        1. A bullish candle
        2. A bearish candle with the same open price but lower close

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 2 candles
        for i in range(1, len(data)):
            # Calculate the percentage difference between opens
            open_diff_pct = abs(open_prices[i] - open_prices[i - 1]) / open_prices[i - 1]

            # Opens should be very close (within 0.1%)
            if open_diff_pct > 0.001:
                continue

            # Check for bullish separating line
            if (close_prices[i - 1] < open_prices[i - 1] and  # First candle bearish
                    close_prices[i] > open_prices[i] and  # Second candle bullish
                    close_prices[i] > close_prices[i - 1]):  # Second close higher than first
                signals[i] = 1

            # Check for bearish separating line
            elif (close_prices[i - 1] > open_prices[i - 1] and  # First candle bullish
                  close_prices[i] < open_prices[i] and  # Second candle bearish
                  close_prices[i] < close_prices[i - 1]):  # Second close lower than first
                signals[i] = -1

        return signals

    @staticmethod
    def detect_falling_three_methods(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Falling Three Methods pattern (bearish continuation).

        The pattern consists of:
        1. A strong bearish candle
        2. Three small bullish candles that stay within the range of the first candle
        3. A final bearish candle that closes below the close of the first candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 5 candles to detect this pattern
        for i in range(4, len(data)):
            # First candle should be a strong bearish candle
            first_bearish = close_prices[i - 4] < open_prices[i - 4]
            first_body = open_prices[i - 4] - close_prices[i - 4]
            first_range = high_prices[i - 4] - low_prices[i - 4]

            strong_first = first_bearish and first_body > 0.6 * first_range

            # Middle three candles should be bullish and small
            middle_bullish = (
                    close_prices[i - 3] > open_prices[i - 3] and
                    close_prices[i - 2] > open_prices[i - 2] and
                    close_prices[i - 1] > open_prices[i - 1]
            )

            # Middle candles should stay within the high-low range of the first candle
            stay_within_range = (
                    high_prices[i - 3] <= high_prices[i - 4] and
                    high_prices[i - 2] <= high_prices[i - 4] and
                    high_prices[i - 1] <= high_prices[i - 4] and
                    low_prices[i - 3] >= low_prices[i - 4] and
                    low_prices[i - 2] >= low_prices[i - 4] and
                    low_prices[i - 1] >= low_prices[i - 4]
            )

            # Small bodies in the middle
            middle_bodies = (
                    abs(close_prices[i - 3] - open_prices[i - 3]) < 0.5 * first_body and
                    abs(close_prices[i - 2] - open_prices[i - 2]) < 0.5 * first_body and
                    abs(close_prices[i - 1] - open_prices[i - 1]) < 0.5 * first_body
            )

            # Last candle should be bearish and close below the first candle's close
            last_bearish = close_prices[i] < open_prices[i]
            closes_lower = close_prices[i] <= close_prices[i - 4]

            if (strong_first and middle_bullish and stay_within_range and
                    middle_bodies and last_bearish and closes_lower):
                signals[i] = -1

        return signals

    @staticmethod
    def detect_doji_star(data: pd.DataFrame, tolerance: float = 0.1) -> np.ndarray:
        """
        Detect Doji Star pattern (potential reversal).

        Bearish Doji Star:
        1. A strong bullish candle in an uptrend
        2. A doji that gaps up from the previous candle

        Bullish Doji Star:
        1. A strong bearish candle in a downtrend
        2. A doji that gaps down from the previous candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            tolerance (float, optional): Maximum body/range ratio for doji. Defaults to 0.1.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 2 candles
        for i in range(1, len(data)):
            # Check if second candle is a doji
            body_size = abs(close_prices[i] - open_prices[i])
            range_size = high_prices[i] - low_prices[i]

            if range_size == 0:
                continue

            is_doji = body_size / range_size < tolerance

            if not is_doji:
                continue

            # Check for bearish doji star
            if (close_prices[i - 1] > open_prices[i - 1] and  # First candle bullish
                    low_prices[i] > close_prices[i - 1]):  # Gap up
                signals[i] = -1

            # Check for bullish doji star
            elif (close_prices[i - 1] < open_prices[i - 1] and  # First candle bearish
                  high_prices[i] < close_prices[i - 1]):  # Gap down
                signals[i] = 1

        return signals

    @staticmethod
    def detect_last_engulfing_top(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Last Engulfing Top pattern (bearish reversal).

        The pattern consists of:
        1. An uptrend
        2. A bullish candle
        3. A bearish candle that completely engulfs the previous candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles to confirm an uptrend
        for i in range(2, len(data)):
            # Check for uptrend
            uptrend = close_prices[i - 2] < close_prices[i - 1]

            # Previous candle should be bullish
            prev_bullish = close_prices[i - 1] > open_prices[i - 1]

            # Current candle should be bearish
            current_bearish = close_prices[i] < open_prices[i]

            # Current candle engulfs previous candle
            engulfing = (
                    open_prices[i] > close_prices[i - 1] and
                    close_prices[i] < open_prices[i - 1]
            )

            if uptrend and prev_bullish and current_bearish and engulfing:
                signals[i] = -1

        return signals

    @staticmethod
    def detect_two_black_gapping(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Two Black Gapping pattern (bearish continuation).

        The pattern consists of:
        1. An uptrend
        2. A gap up
        3. Two consecutive bearish candles

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # Check for gap up
            gap_up = low_prices[i - 1] > high_prices[i - 2]

            # Check for two consecutive bearish candles
            both_bearish = (
                    close_prices[i - 1] < open_prices[i - 1] and
                    close_prices[i] < open_prices[i]
            )

            # Second bearish candle should open near the first bearish candle's open
            similar_opens = abs(open_prices[i] - open_prices[i - 1]) / open_prices[i - 1] < 0.01

            if gap_up and both_bearish and similar_opens:
                signals[i] = -1

        return signals

    @staticmethod
    def detect_side_by_side_white_lines(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Side by Side White Lines pattern (bullish continuation).

        The pattern consists of:
        1. A strong bullish candle
        2. A gap up
        3. Two bullish candles with similar size and position

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # First candle should be bullish
            first_bullish = close_prices[i - 2] > open_prices[i - 2]

            # Check for gap up
            gap_up = low_prices[i - 1] > high_prices[i - 2]

            # Both second and third candles should be bullish
            both_bullish = (
                    close_prices[i - 1] > open_prices[i - 1] and
                    close_prices[i] > open_prices[i]
            )

            # Similar body sizes and positions for second and third candles
            second_body = close_prices[i - 1] - open_prices[i - 1]
            third_body = close_prices[i] - open_prices[i]

            # Check if second_body is zero to avoid division by zero
            if second_body == 0:
                # If second_body is zero, only consider it similar if third_body is also very small
                similar_body_size = third_body < 0.01  # Some small threshold
            else:
                similar_body_size = abs(third_body - second_body) / second_body < 0.2

            # Check for open_prices[i-1] being zero to avoid division by zero
            if open_prices[i - 1] == 0 or close_prices[i - 1] == 0:
                similar_positions = False
            else:
                similar_positions = (
                        abs(open_prices[i] - open_prices[i - 1]) / open_prices[i - 1] < 0.02 and
                        abs(close_prices[i] - close_prices[i - 1]) / close_prices[i - 1] < 0.02
                )

            if first_bullish and gap_up and both_bullish and similar_body_size and similar_positions:
                signals[i] = 1

        return signals

    @staticmethod
    def detect_three_stars_in_the_south(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Three Stars in the South pattern (bullish reversal).

        The pattern consists of:
        1. A long bearish candle
        2. A smaller bearish candle with a lower low but higher close than the first
        3. A small bearish candle with a higher close than the second

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # All three candles should be bearish
            all_bearish = (
                    close_prices[i - 2] < open_prices[i - 2] and
                    close_prices[i - 1] < open_prices[i - 1] and
                    close_prices[i] < open_prices[i]
            )

            if not all_bearish:
                continue

            # First candle should be long
            first_body = open_prices[i - 2] - close_prices[i - 2]
            first_range = high_prices[i - 2] - low_prices[i - 2]

            long_first = first_body > 0.6 * first_range

            # Second candle should have lower low but higher close
            second_body = open_prices[i - 1] - close_prices[i - 1]
            lower_low = low_prices[i - 1] < low_prices[i - 2]
            higher_close = close_prices[i - 1] > close_prices[i - 2]

            # Second candle should be smaller than first
            smaller_second = second_body < first_body

            # Third candle should be small with higher close
            third_body = open_prices[i] - close_prices[i]
            small_third = third_body < second_body
            third_higher_close = close_prices[i] > close_prices[i - 1]

            if (long_first and lower_low and higher_close and smaller_second and
                    small_third and third_higher_close):
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_three_line_strike(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Three Line Strike pattern (bullish reversal after bearish trend).

        The pattern consists of:
        1. Three consecutive bearish candles, each with a lower low and lower close
        2. A fourth bullish candle that opens below the third candle and closes above
           the first candle's open

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 4 candles
        for i in range(3, len(data)):
            # First three candles should be bearish with lower lows and lower closes
            three_bearish = (
                    close_prices[i - 3] < open_prices[i - 3] and
                    close_prices[i - 2] < open_prices[i - 2] and
                    close_prices[i - 1] < open_prices[i - 1]
            )

            lower_lows = (
                    low_prices[i - 2] < low_prices[i - 3] and
                    low_prices[i - 1] < low_prices[i - 2]
            )

            lower_closes = (
                    close_prices[i - 2] < close_prices[i - 3] and
                    close_prices[i - 1] < close_prices[i - 2]
            )

            # Fourth candle should be bullish, open below third and close above first
            fourth_bullish = close_prices[i] > open_prices[i]
            opens_below = open_prices[i] < close_prices[i - 1]
            closes_above = close_prices[i] > open_prices[i - 3]

            if (three_bearish and lower_lows and lower_closes and
                    fourth_bullish and opens_below and closes_above):
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_identical_three_crows(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Identical Three Crows pattern (bearish reversal).

        The pattern consists of:
        1. Three consecutive bearish candles with similar size and shape
        2. Each opening near the previous candle's close
        3. Each closing near its low

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # All three candles should be bearish
            all_bearish = (
                    close_prices[i - 2] < open_prices[i - 2] and
                    close_prices[i - 1] < open_prices[i - 1] and
                    close_prices[i] < open_prices[i]
            )

            if not all_bearish:
                continue

            # Calculate body sizes
            first_body = open_prices[i - 2] - close_prices[i - 2]
            second_body = open_prices[i - 1] - close_prices[i - 1]
            third_body = open_prices[i] - close_prices[i]

            # Check for similar body sizes
            similar_sizes = (
                    abs(second_body - first_body) / first_body < 0.2 and
                    abs(third_body - second_body) / second_body < 0.2
            )

            # Each opens near the previous close
            opens_near_close = (
                    abs(open_prices[i - 1] - close_prices[i - 2]) / close_prices[i - 2] < 0.02 and
                    abs(open_prices[i] - close_prices[i - 1]) / close_prices[i - 1] < 0.02
            )

            # Each closes near its low
            closes_near_low = (
                    (close_prices[i - 2] - low_prices[i - 2]) / (high_prices[i - 2] - low_prices[i - 2]) < 0.1 and
                    (close_prices[i - 1] - low_prices[i - 1]) / (high_prices[i - 1] - low_prices[i - 1]) < 0.1 and
                    (close_prices[i] - low_prices[i]) / (high_prices[i] - low_prices[i]) < 0.1
            )

            if all_bearish and similar_sizes and opens_near_close and closes_near_low:
                signals[i] = -1  # Bearish reversal signal

        return signals

    @staticmethod
    def detect_morning_doji_star(data: pd.DataFrame, doji_tolerance: float = 0.1) -> np.ndarray:
        """
        Detect Morning Doji Star pattern (bullish reversal).

        The pattern consists of:
        1. A bearish candle
        2. A doji that gaps down
        3. A bullish candle that gaps up and closes into the first candle's body

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            doji_tolerance (float, optional): Maximum body/range ratio for doji. Defaults to 0.1.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # First candle should be bearish
            first_bearish = close_prices[i - 2] < open_prices[i - 2]

            # Second candle should be a doji
            second_body = abs(close_prices[i - 1] - open_prices[i - 1])
            second_range = high_prices[i - 1] - low_prices[i - 1]

            is_doji = (second_range > 0 and second_body / second_range < doji_tolerance)

            # Doji should gap down
            gap_down = high_prices[i - 1] < close_prices[i - 2]

            # Third candle should be bullish
            third_bullish = close_prices[i] > open_prices[i]

            # Third candle should gap up from doji
            gap_up = open_prices[i] > close_prices[i - 1]

            # Third candle should close well into first candle's body
            first_body_mid = (open_prices[i - 2] + close_prices[i - 2]) / 2
            closes_into_first = close_prices[i] >= first_body_mid

            if (first_bearish and is_doji and gap_down and
                    third_bullish and gap_up and closes_into_first):
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_three_outside_up(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Three Outside Up pattern (bullish reversal).

        The pattern consists of:
        1. A bearish candle
        2. A bullish candle that engulfs the first candle
        3. A third bullish candle that closes higher

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # First candle should be bearish
            first_bearish = close_prices[i - 2] < open_prices[i - 2]

            # Second candle should be bullish and engulf the first
            second_bullish = close_prices[i - 1] > open_prices[i - 1]
            engulfs_first = (
                    open_prices[i - 1] < close_prices[i - 2] and
                    close_prices[i - 1] > open_prices[i - 2]
            )

            # Third candle should be bullish with higher close
            third_bullish = close_prices[i] > open_prices[i]
            higher_close = close_prices[i] > close_prices[i - 1]

            if first_bearish and second_bullish and engulfs_first and third_bullish and higher_close:
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_three_line_strike_bearish(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Three Line Strike Bearish pattern (bullish reversal after bearish trend).

        The pattern consists of:
        1. Three consecutive bearish candles, each with a lower low and lower close
        2. A fourth bullish candle that opens below the third candle and closes above
           the first candle's open

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 4 candles
        for i in range(3, len(data)):
            # First three candles should be bearish with lower lows and lower closes
            three_bearish = (
                    close_prices[i - 3] < open_prices[i - 3] and
                    close_prices[i - 2] < open_prices[i - 2] and
                    close_prices[i - 1] < open_prices[i - 1]
            )

            lower_lows = (
                    low_prices[i - 2] < low_prices[i - 3] and
                    low_prices[i - 1] < low_prices[i - 2]
            )

            lower_closes = (
                    close_prices[i - 2] < close_prices[i - 3] and
                    close_prices[i - 1] < close_prices[i - 2]
            )

            # Fourth candle should be bullish, open below third and close above first
            fourth_bullish = close_prices[i] > open_prices[i]
            opens_below = open_prices[i] < close_prices[i - 1]
            closes_above = close_prices[i] > open_prices[i - 3]

            if (three_bearish and lower_lows and lower_closes and
                    fourth_bullish and opens_below and closes_above):
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_three_line_strike_bullish(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Three Line Strike Bullish pattern (bearish reversal after bullish trend).

        The pattern consists of:
        1. Three consecutive bullish candles, each with a higher high and higher close
        2. A fourth bearish candle that opens above the third candle and closes below
           the first candle's open

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 4 candles
        for i in range(3, len(data)):
            # First three candles should be bullish with higher highs and higher closes
            three_bullish = (
                    close_prices[i - 3] > open_prices[i - 3] and
                    close_prices[i - 2] > open_prices[i - 2] and
                    close_prices[i - 1] > open_prices[i - 1]
            )

            higher_highs = (
                    high_prices[i - 2] > high_prices[i - 3] and
                    high_prices[i - 1] > high_prices[i - 2]
            )

            higher_closes = (
                    close_prices[i - 2] > close_prices[i - 3] and
                    close_prices[i - 1] > close_prices[i - 2]
            )

            # Fourth candle should be bearish, open above third and close below first
            fourth_bearish = close_prices[i] < open_prices[i]
            opens_above = open_prices[i] > close_prices[i - 1]
            closes_below = close_prices[i] < open_prices[i - 3]

            if (three_bullish and higher_highs and higher_closes and
                    fourth_bearish and opens_above and closes_below):
                signals[i] = -1  # Bearish reversal signal

        return signals

    @staticmethod
    def detect_upside_tasuki_gap(data: pd.DataFrame) -> np.ndarray:
        """
        Detect Upside Tasuki Gap pattern (bullish continuation).

        The pattern consists of:
        1. A bullish candle
        2. A gap up followed by another bullish candle
        3. A bearish candle that opens within the body of the second candle
           and closes within the gap but does not close it completely

        Args:
            data (pd.DataFrame): Market data with OHLC prices.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # First two candles should be bullish
            first_bullish = close_prices[i - 2] > open_prices[i - 2]
            second_bullish = close_prices[i - 1] > open_prices[i - 1]

            # Check for gap up between first and second candles
            gap_up = low_prices[i - 1] > high_prices[i - 2]

            # Third candle should be bearish
            third_bearish = close_prices[i] < open_prices[i]

            # Third candle opens within second candle's body
            opens_within = (
                    open_prices[i] > open_prices[i - 1] and
                    open_prices[i] < close_prices[i - 1]
            )

            # Third candle closes within the gap but doesn't close it completely
            closes_in_gap = (
                    close_prices[i] < low_prices[i - 1] and
                    close_prices[i] > high_prices[i - 2]
            )

            if (first_bullish and second_bullish and gap_up and
                    third_bearish and opens_within and closes_in_gap):
                signals[i] = 1  # Bullish continuation signal

        return signals

    @staticmethod
    def detect_hammer_inverted(data: pd.DataFrame, body_threshold: float = 0.3,
                               upper_shadow_threshold: float = 0.6,
                               lower_shadow_threshold: float = 0.1) -> np.ndarray:
        """
        Detect Inverted Hammer pattern (bearish continuation in a downtrend).

        An inverted hammer has a small body at the lower end of the trading range
        with a long upper shadow and little or no lower shadow.

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            body_threshold (float, optional): Maximum body/range ratio. Defaults to 0.3.
            upper_shadow_threshold (float, optional): Minimum upper shadow/range ratio. Defaults to 0.6.
            lower_shadow_threshold (float, optional): Maximum lower shadow/range ratio. Defaults to 0.1.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Calculate body size and total range
            body_size = abs(close_prices[i] - open_prices[i])
            total_range = high_prices[i] - low_prices[i]

            # Avoid division by zero
            if total_range == 0:
                continue

            body_ratio = body_size / total_range

            # Calculate shadows
            if close_prices[i] >= open_prices[i]:  # Bullish candle
                upper_shadow = high_prices[i] - close_prices[i]
                lower_shadow = open_prices[i] - low_prices[i]
            else:  # Bearish candle
                upper_shadow = high_prices[i] - open_prices[i]
                lower_shadow = close_prices[i] - low_prices[i]

            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range

            # Check if we have an inverted hammer
            is_inverted_hammer = (
                    body_ratio <= body_threshold and
                    upper_shadow_ratio >= upper_shadow_threshold and
                    lower_shadow_ratio <= lower_shadow_threshold
            )

            # Check for downtrend
            in_downtrend = i > 1 and close_prices[i - 1] < close_prices[i - 2]

            if is_inverted_hammer and in_downtrend:
                signals[i] = -1  # Bearish continuation signal

        return signals

    @staticmethod
    def detect_matching_low(data: pd.DataFrame, threshold: float = 0.003) -> np.ndarray:
        """
        Detect Matching Low pattern (bearish continuation).

        The pattern consists of:
        1. A bearish candle in a downtrend
        2. Another bearish candle with a close very close to the previous close

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            threshold (float, optional): Maximum difference between closes. Defaults to 0.003.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles to confirm downtrend
        for i in range(2, len(data)):
            # Check for downtrend
            downtrend = close_prices[i - 2] > close_prices[i - 1]

            # Both candles should be bearish
            both_bearish = (
                    close_prices[i - 1] < open_prices[i - 1] and
                    close_prices[i] < open_prices[i]
            )

            # Closes should be very close to each other
            matching_closes = (
                    abs(close_prices[i] - close_prices[i - 1]) / close_prices[i - 1] <= threshold
            )

            if downtrend and both_bearish and matching_closes:
                signals[i] = -1  # Bearish continuation signal

        return signals

    @staticmethod
    def detect_abandoned_baby(data: pd.DataFrame, doji_threshold: float = 0.1,
                              gap_threshold: float = 0.001) -> np.ndarray:
        """
        Detect Abandoned Baby pattern (bullish reversal).

        The pattern consists of:
        1. A bearish candle
        2. A doji that gaps down and is isolated from the candles on either side
        3. A bullish candle that gaps up from the doji

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            doji_threshold (float, optional): Maximum body/range ratio for doji. Defaults to 0.1.
            gap_threshold (float, optional): Minimum gap size as percentage. Defaults to 0.001.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 3 candles
        for i in range(2, len(data)):
            # First candle should be bearish
            first_bearish = close_prices[i - 2] < open_prices[i - 2]

            # Second candle should be a doji
            second_body = abs(close_prices[i - 1] - open_prices[i - 1])
            second_range = high_prices[i - 1] - low_prices[i - 1]

            is_doji = second_range > 0 and second_body / second_range <= doji_threshold

            # Check for gaps on both sides of the doji
            gap_down = high_prices[i - 1] < low_prices[i - 2]
            gap_up = low_prices[i] > high_prices[i - 1]

            # Third candle should be bullish
            third_bullish = close_prices[i] > open_prices[i]

            if first_bearish and is_doji and gap_down and gap_up and third_bullish:
                signals[i] = 1  # Bullish reversal signal

        return signals

    @staticmethod
    def detect_breakaway_bearish(data: pd.DataFrame, gap_threshold: float = 0.01) -> np.ndarray:
        """
        Detect Bearish Breakaway pattern (bearish reversal).

        The pattern consists of:
        1. A long bullish candle in an uptrend
        2. A gap up to the second bullish candle
        3. The third and possibly fourth candles continue with small bullish candles
        4. The final candle is a long bearish candle that closes well into the body of the first candle

        Args:
            data (pd.DataFrame): Market data with OHLC prices.
            gap_threshold (float, optional): Minimum size for gaps. Defaults to 0.01.

        Returns:
            np.ndarray: Array of pattern signals.
        """
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        signals = np.zeros(len(data))

        # Need at least 5 candles
        for i in range(4, len(data)):
            # First candle should be a strong bullish candle
            first_bullish = close_prices[i - 4] > open_prices[i - 4]
            first_body = close_prices[i - 4] - open_prices[i - 4]
            first_range = high_prices[i - 4] - low_prices[i - 4]
            strong_first = first_bullish and first_body > 0.6 * first_range

            # Gap up to second candle
            gap_up = low_prices[i - 3] > high_prices[i - 4]

            # Second and third candles should be bullish (can be small)
            middle_bullish = (
                    close_prices[i - 3] > open_prices[i - 3] and
                    close_prices[i - 2] > open_prices[i - 2]
            )

            # Fourth candle can be bullish or bearish

            # Final candle should be strongly bearish and close well into first candle
            last_bearish = close_prices[i] < open_prices[i]
            last_body = open_prices[i] - close_prices[i]
            strong_last = last_bearish and last_body > 0.5 * first_body

            # Final close should penetrate into the first candle's body
            closes_into_first = close_prices[i] < ((open_prices[i - 4] + close_prices[i - 4]) / 2)

            if (strong_first and gap_up and middle_bullish and
                    last_bearish and strong_last and closes_into_first):
                signals[i] = -1  # Bearish reversal signal

        return signals

        