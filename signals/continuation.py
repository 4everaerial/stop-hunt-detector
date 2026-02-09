"""
Continuation failure signal.

Measures momentum vs price divergence. When momentum fades but price holds,
markets are fragile.
"""

import pandas as pd
import numpy as np


def calculate_rsi(df, period=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with close prices
        period: RSI period

    Returns:
        Series with RSI values
    """
    close = df['close']

    # Calculate price changes
    delta = close.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()

    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_rsi_divergence(df, period=14):
    """
    Calculate RSI divergence.

    Compares price trend to RSI trend. Divergence indicates weakening momentum.

    Args:
        df: DataFrame with OHLCV data
        period: RSI period

    Returns:
        Series with divergence score (0-1, higher = more divergence)
    """
    rsi = calculate_rsi(df, period)
    close = df['close']

    # Calculate trend over 20 periods
    price_trend = close.pct_change(20)
    rsi_trend = rsi.diff(20)

    # Divergence: opposite trends
    divergence = abs(price_trend * rsi_trend)

    # Normalize
    divergence_norm = divergence / divergence.rolling(window=50).max()

    return divergence_norm


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame with close prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Tuple of (macd, signal, histogram)
    """
    close = df['close']

    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    # Calculate MACD line
    macd = ema_fast - ema_slow

    # Calculate signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    # Calculate histogram
    histogram = macd - signal_line

    return macd, signal_line, histogram


def calculate_macd_divergence(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD divergence.

    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Series with divergence score (0-1, higher = more divergence)
    """
    macd, signal_line, _ = calculate_macd(df, fast, slow, signal)
    close = df['close']

    # Calculate trend over 20 periods
    price_trend = close.pct_change(20)
    macd_trend = macd.diff(20)

    # Divergence: opposite trends
    divergence = abs(price_trend * macd_trend)

    # Normalize
    divergence_norm = divergence / divergence.rolling(window=50).max()

    return divergence_norm


def calculate_volume_price_divergence(df, period=20):
    """
    Calculate volume vs price divergence.

    When volume trends opposite to price, indicates weak continuation.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        Series with divergence score (0-1, higher = more divergence)
    """
    price_trend = df['close'].pct_change(period)
    volume_trend = df['volume'].pct_change(period)

    # Divergence: opposite trends
    divergence = abs(price_trend * volume_trend)

    # Normalize
    divergence_norm = divergence / divergence.rolling(window=50).max()

    return divergence_norm


class ContinuationSignal:
    """Calculate continuation failure stress signal."""

    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, period=20):
        """
        Initialize continuation signal.

        Args:
            rsi_period: RSI period
            macd_fast: MACD fast EMA
            macd_slow: MACD slow EMA
            macd_signal: MACD signal period
            period: Divergence lookback period
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.period = period

    def calculate(self, df):
        """
        Calculate continuation failure signal.

        Returns DataFrame with new columns:
        - rsi_divergence: RSI divergence score
        - macd_divergence: MACD divergence score
        - volume_divergence: Volume-price divergence score
        - continuation_score: Normalized stress score (0-1)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with continuation signals
        """
        df = df.copy()

        # Calculate individual indicators
        df['rsi_divergence'] = calculate_rsi_divergence(df, self.rsi_period)
        df['macd_divergence'] = calculate_macd_divergence(
            df, self.macd_fast, self.macd_slow, self.macd_signal
        )
        df['volume_divergence'] = calculate_volume_price_divergence(df, self.period)

        # Composite continuation score (equal weights)
        df['continuation_score'] = (
            df['rsi_divergence'].fillna(0) * 0.34 +
            df['macd_divergence'].fillna(0) * 0.33 +
            df['volume_divergence'].fillna(0) * 0.33
        )

        # Clip to 0-1 range
        df['continuation_score'] = df['continuation_score'].clip(0, 1)

        return df


# Backward compatibility
def calculate_continuation_signal(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, period=20):
    """
    Calculate continuation failure signal (functional API).

    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI period
        macd_fast: MACD fast EMA
        macd_slow: MACD slow EMA
        macd_signal: MACD signal period
        period: Divergence lookback period

    Returns:
        DataFrame with continuation signals
    """
    signal = ContinuationSignal(rsi_period, macd_fast, macd_slow, macd_signal, period)
    return signal.calculate(df)
