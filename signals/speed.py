"""
Speed asymmetry signal.

Measures down-move vs up-move velocity. Panic selling is faster than
panic buying.
"""

import pandas as pd
import numpy as np


def calculate_down_velocity(df, period=5):
    """
    Calculate down-move velocity.

    Measures how fast price drops during down candles.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        Series with down velocity (same index as input df)
    """
    # Calculate price change
    close = df['close']

    # Calculate price change
    change = close.pct_change(period).abs()

    # Filter down candles (close < open)
    is_down = df['close'] < df['open']

    # Set non-down candles to NaN
    down_change = change.where(is_down)

    return down_change


def calculate_up_velocity(df, period=5):
    """
    Calculate up-move velocity.

    Measures how fast price rises during up candles.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        Series with up velocity (same index as input df)
    """
    # Calculate price change
    close = df['close']

    # Calculate price change
    change = close.pct_change(period)

    # Filter up candles (close > open)
    is_up = df['close'] > df['open']

    # Set non-up candles to NaN
    up_change = change.where(is_up)

    return up_change


def calculate_velocity_asymmetry(df, period=5):
    """
    Calculate velocity asymmetry (down vs up).

    Panic selling is typically faster than panic buying.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        Series with asymmetry score (0-1, higher = more asymmetry)
    """
    # Calculate absolute price changes
    close = df['close']
    change = close.pct_change(period).abs()

    # Separate down and up candle changes
    is_down = df['close'] < df['open']
    is_up = df['close'] > df['open']

    down_change = change.where(is_down, 0)
    up_change = change.where(is_up, 0)

    # Calculate rolling sums
    down_sum = down_change.rolling(window=period).sum()
    up_sum = up_change.rolling(window=period).sum()

    # Asymmetry ratio: down / (up + epsilon)
    asymmetry = down_sum / (up_sum + 1e-6)

    # Normalize (clipping outliers)
    asymmetry = asymmetry.clip(0, 5) / 5

    return asymmetry


def calculate_drawdown_speed(df, period=20):
    """
    Calculate maximum drawdown speed.

    Measures how fast price drops from recent highs.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        Series with drawdown speed (0-1, higher = faster drawdown)
    """
    close = df['close']

    # Calculate rolling maximum
    rolling_max = close.rolling(window=period).max()

    # Calculate drawdown
    drawdown = (rolling_max - close) / rolling_max

    # Calculate drawdown speed (change in drawdown)
    drawdown_speed = drawdown.diff(period).abs()

    # Normalize
    drawdown_speed_norm = drawdown_speed / drawdown_speed.rolling(window=50).max()

    return drawdown_speed_norm


def calculate_volume_weighted_direction(df, period=10):
    """
    Calculate volume-weighted price direction.

    Down moves with high volume indicate panic selling.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        Series with direction score (0-1, negative + high volume = high stress)
    """
    # Calculate price change
    change = df['close'].pct_change()

    # Weight by volume
    volume_change = change * df['volume']

    # Calculate rolling sum
    weighted_change = volume_change.rolling(window=period).sum()

    # Normalize: negative = more stress
    direction_score = (-weighted_change).clip(lower=0)
    direction_score = direction_score / direction_score.rolling(window=50).max()

    return direction_score


class SpeedSignal:
    """Calculate speed asymmetry stress signal."""

    def __init__(self, velocity_period=5, drawdown_period=20, direction_period=10):
        """
        Initialize speed signal.

        Args:
            velocity_period: Velocity lookback period
            drawdown_period: Drawdown lookback period
            direction_period: Volume-weighted direction lookback
        """
        self.velocity_period = velocity_period
        self.drawdown_period = drawdown_period
        self.direction_period = direction_period

    def calculate(self, df):
        """
        Calculate speed asymmetry signal.

        Returns DataFrame with new columns:
        - velocity_asymmetry: Down vs up velocity ratio
        - drawdown_speed: Maximum drawdown speed
        - volume_direction: Volume-weighted direction score
        - speed_score: Normalized stress score (0-1)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with speed signals
        """
        df = df.copy()

        # Calculate individual indicators
        df['velocity_asymmetry'] = calculate_velocity_asymmetry(df, self.velocity_period)
        df['drawdown_speed'] = calculate_drawdown_speed(df, self.drawdown_period)
        df['volume_direction'] = calculate_volume_weighted_direction(df, self.direction_period)

        # Composite speed score (equal weights)
        df['speed_score'] = (
            df['velocity_asymmetry'].fillna(0) * 0.34 +
            df['drawdown_speed'].fillna(0) * 0.33 +
            df['volume_direction'].fillna(0) * 0.33
        )

        # Clip to 0-1 range
        df['speed_score'] = df['speed_score'].clip(0, 1)

        return df


# Backward compatibility
def calculate_speed_signal(df, velocity_period=5, drawdown_period=20, direction_period=10):
    """
    Calculate speed asymmetry signal (functional API).

    Args:
        df: DataFrame with OHLCV data
        velocity_period: Velocity lookback period
        drawdown_period: Drawdown lookback period
        direction_period: Volume-weighted direction lookback

    Returns:
        DataFrame with speed signals
    """
    signal = SpeedSignal(velocity_period, drawdown_period, direction_period)
    return signal.calculate(df)
