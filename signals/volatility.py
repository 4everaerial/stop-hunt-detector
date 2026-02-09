"""
Volatility compression signal.

Measures range tightening and ATR decay. Volatility compression often precedes
sudden market moves.
"""

import pandas as pd
import numpy as np


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with OHLC data
        period: ATR period

    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using RMA (Rolling Moving Average)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame with close prices
        period: Moving average period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band, bandwidth)
    """
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    # Bandwidth: (upper - lower) / middle
    bandwidth = (upper - lower) / middle

    return upper, middle, lower, bandwidth


def calculate_range_compression(df, period=20):
    """
    Calculate price range compression.

    Compares current high-low range to average range over period.

    Args:
        df: DataFrame with OHLC data
        period: Lookback period

    Returns:
        Series with range compression ratio (0-1, lower = more compressed)
    """
    current_range = df['high'] - df['low']
    avg_range = current_range.rolling(window=period).mean()

    # Compression ratio: current / avg
    # Lower values indicate more compression
    compression = current_range / avg_range

    return compression


def calculate_atr_decay(df, short_period=10, long_period=50):
    """
    Calculate ATR decay rate.

    Compares short-term ATR to long-term ATR. Lower ratio indicates
    volatility compression.

    Args:
        df: DataFrame with OHLC data
        short_period: Short ATR period
        long_period: Long ATR period

    Returns:
        Series with ATR decay ratio (0-1, lower = more decay)
    """
    short_atr = calculate_atr(df, short_period)
    long_atr = calculate_atr(df, long_period)

    decay = short_atr / long_atr

    return decay


class VolatilitySignal:
    """Calculate volatility compression stress signal."""

    def __init__(self, atr_short=10, atr_long=50, bb_period=20, range_period=20):
        """
        Initialize volatility signal.

        Args:
            atr_short: Short-term ATR period
            atr_long: Long-term ATR period
            bb_period: Bollinger Bands period
            range_period: Range compression lookback
        """
        self.atr_short = atr_short
        self.atr_long = atr_long
        self.bb_period = bb_period
        self.range_period = range_period

    def calculate(self, df):
        """
        Calculate volatility compression signal.

        Returns DataFrame with new columns:
        - atr_short: Short-term ATR
        - atr_long: Long-term ATR
        - atr_decay: ATR decay ratio
        - bb_bandwidth: Bollinger Band width
        - range_compression: Price range compression
        - volatility_score: Normalized stress score (0-1)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility signals
        """
        df = df.copy()

        # Calculate ATRs
        df['atr_short'] = calculate_atr(df, self.atr_short)
        df['atr_long'] = calculate_atr(df, self.atr_long)
        df['atr_decay'] = calculate_atr_decay(df, self.atr_short, self.atr_long)

        # Calculate Bollinger Bands
        upper, middle, lower, bandwidth = calculate_bollinger_bands(
            df, self.bb_period
        )
        df['bb_bandwidth'] = bandwidth

        # Calculate range compression
        df['range_compression'] = calculate_range_compression(df, self.range_period)

        # Normalize indicators to 0-1 (inverted: higher = more stress)
        # ATR decay: lower ratio = more compression = more stress
        df['atr_decay_norm'] = 1 - df['atr_decay'].clip(0, 1)

        # BB bandwidth: lower width = more compression = more stress
        bb_median = df['bb_bandwidth'].rolling(window=50).median()
        df['bb_bandwidth_norm'] = (bb_median - df['bb_bandwidth']).clip(lower=0)
        df['bb_bandwidth_norm'] = df['bb_bandwidth_norm'] / df['bb_bandwidth_norm'].max()

        # Range compression: lower ratio = more compression = more stress
        df['range_compression_norm'] = 1 - df['range_compression'].clip(0, 1)

        # Composite volatility score (equal weights)
        df['volatility_score'] = (
            df['atr_decay_norm'].fillna(0) * 0.34 +
            df['bb_bandwidth_norm'].fillna(0) * 0.33 +
            df['range_compression_norm'].fillna(0) * 0.33
        )

        # Clip to 0-1 range
        df['volatility_score'] = df['volatility_score'].clip(0, 1)

        return df


# Backward compatibility
def calculate_volatility_signal(df, atr_short=10, atr_long=50, bb_period=20, range_period=20):
    """
    Calculate volatility compression signal (functional API).

    Args:
        df: DataFrame with OHLCV data
        atr_short: Short-term ATR period
        atr_long: Long-term ATR period
        bb_period: Bollinger Bands period
        range_period: Range compression lookback

    Returns:
        DataFrame with volatility signals
    """
    signal = VolatilitySignal(atr_short, atr_long, bb_period, range_period)
    return signal.calculate(df)
