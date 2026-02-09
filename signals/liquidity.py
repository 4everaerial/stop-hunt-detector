"""
Liquidity fragility signal.

Measures wick-to-body ratio and reaction magnitude. Thin liquidity leads to
exaggerated price moves.
"""

import pandas as pd
import numpy as np


def calculate_wick_body_ratio(df):
    """
    Calculate wick-to-body ratio.

    Compares upper + lower wicks to candle body. Higher ratios indicate
    thin liquidity.

    Args:
        df: DataFrame with OHLC data

    Returns:
        Series with wick-to-body ratio (0-1, higher = more fragile)
    """
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']

    # Calculate body and wicks
    body = abs(close - open_)
    body = body.replace(0, np.nan)  # Avoid division by zero

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    # Wick-to-body ratio
    ratio = (upper_wick + lower_wick) / body

    # Normalize (clipping outliers)
    ratio = ratio.clip(0, 5) / 5

    return ratio


def calculate_order_book_imbalance(df):
    """
    Calculate order book imbalance proxy.

    Uses volume on up vs down candles to estimate liquidity balance.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Series with imbalance score (0-1, extreme = more fragile)
    """
    # Up candle (close > open) volume vs down candle volume
    df = df.copy()

    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']

    up_volume = df[is_up]['volume'].rolling(window=10).sum()
    down_volume = df[is_down]['volume'].rolling(window=10).sum()

    # Imbalance ratio
    total = up_volume + down_volume
    imbalance = abs(up_volume - down_volume) / total

    return imbalance


def calculate_reaction_magnitude(df, period=5):
    """
    Calculate reaction magnitude after moves.

    Measures how much price reverses after a move, indicating
    thin liquidity.

    Args:
        df: DataFrame with OHLC data
        period: Lookback period

    Returns:
        Series with reaction magnitude (0-1, higher = more fragile)
    """
    # Calculate price change
    change = df['close'].pct_change()

    # Calculate reversal magnitude
    reversal = abs(change - change.shift(1))

    # Normalize
    reversal_norm = reversal / reversal.rolling(window=period).max()

    return reversal_norm


class LiquiditySignal:
    """Calculate liquidity fragility stress signal."""

    def __init__(self):
        """Initialize liquidity signal."""
        pass

    def calculate(self, df):
        """
        Calculate liquidity fragility signal.

        Returns DataFrame with new columns:
        - wick_body_ratio: Wick-to-body ratio
        - order_imbalance: Order book imbalance proxy
        - reaction_magnitude: Reaction magnitude
        - liquidity_score: Normalized stress score (0-1)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with liquidity signals
        """
        df = df.copy()

        # Calculate individual indicators
        df['wick_body_ratio'] = calculate_wick_body_ratio(df)
        df['order_imbalance'] = calculate_order_book_imbalance(df)
        df['reaction_magnitude'] = calculate_reaction_magnitude(df)

        # Composite liquidity score (equal weights)
        df['liquidity_score'] = (
            df['wick_body_ratio'].fillna(0) * 0.34 +
            df['order_imbalance'].fillna(0) * 0.33 +
            df['reaction_magnitude'].fillna(0) * 0.33
        )

        # Clip to 0-1 range
        df['liquidity_score'] = df['liquidity_score'].clip(0, 1)

        return df


# Backward compatibility
def calculate_liquidity_signal(df):
    """
    Calculate liquidity fragility signal (functional API).

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with liquidity signals
    """
    signal = LiquiditySignal()
    return signal.calculate(df)
