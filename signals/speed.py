"""
Speed Asymmetry Signal

Detects when downward price moves are faster and stronger than
upward moves - a hallmark of panic selling and liquidations.

Mechanism:
- Downward velocity vs upward velocity
- Downward magnitude vs upward magnitude
- Negative candle acceleration

Score: 0.0 (balanced) → 1.0 (strongly biased downward)
"""

import pandas as pd
import numpy as np
from typing import Tuple


class SpeedAsymmetry:
    """
    Measures speed asymmetry between up and down moves.

    High speed asymmetry indicates panic selling and
    liquidation-driven downward pressure.
    """

    def __init__(
        self,
        lookback: int = 20,
        velocity_window: int = 5
    ):
        """
        Initialize speed asymmetry detector.

        Args:
            lookback: Lookback period for normalization (default: 20)
            velocity_window: Window for velocity calculation (default: 5)
        """
        self.lookback = lookback
        self.velocity_window = velocity_window

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate speed asymmetry score for each candle.

        Args:
            df: DataFrame with OHLCV data (columns: open, close)

        Returns:
            Series of speed asymmetry scores (0.0 → 1.0)
        """
        df = df.copy()

        # Classify candles as up or down
        df['is_up'] = df['close'] > df['open']
        df['is_down'] = df['close'] < df['open']

        # Calculate price change magnitude
        df['change'] = df['close'] - df['open']
        df['change_abs'] = df['change'].abs()

        # Calculate velocity (change per candle)
        df['velocity'] = df['change']

        # Separate up and down moves
        df['down_change'] = np.where(df['is_down'], df['change_abs'], 0)
        df['up_change'] = np.where(df['is_up'], df['change_abs'], 0)

        df['down_velocity'] = np.where(df['is_down'], df['change'], 0)
        df['up_velocity'] = np.where(df['is_up'], df['change'], 0)

        # Rolling metrics
        down_magnitude = df['down_change'].rolling(self.lookback).sum()
        up_magnitude = df['up_change'].rolling(self.lookback).sum()

        down_velocity_avg = df['down_velocity'].rolling(self.lookback).mean()
        up_velocity_avg = df['up_velocity'].rolling(self.lookback).mean()

        # Calculate asymmetry ratios
        df['magnitude_asymmetry'] = self._calculate_asymmetry_ratio(
            down_magnitude,
            up_magnitude
        )

        df['velocity_asymmetry'] = self._calculate_asymmetry_ratio(
            down_velocity_avg.abs(),
            up_velocity_avg.abs()
        )

        # Negative candle acceleration (increasing down candles)
        df['down_candle_count'] = df['is_down'].rolling(self.velocity_window).sum()
        df['up_candle_count'] = df['is_up'].rolling(self.velocity_window).sum()

        df['candle_asymmetry'] = self._calculate_asymmetry_ratio(
            df['down_candle_count'],
            df['up_candle_count']
        )

        # Composite score (all normalized 0-1)
        df['speed_asymmetry'] = (
            df['magnitude_asymmetry'] * 0.4 +
            df['velocity_asymmetry'] * 0.4 +
            df['candle_asymmetry'] * 0.2
        )

        return df['speed_asymmetry']

    def _calculate_asymmetry_ratio(
        self,
        down_metric: pd.Series,
        up_metric: pd.Series
    ) -> pd.Series:
        """
        Calculate asymmetry ratio (down / total).

        Returns 0.5 when balanced, approaches 1.0 when heavily down.
        """
        total = down_metric + up_metric

        # Avoid division by zero
        ratio = np.where(
            total > 0,
            down_metric / total,
            0.5  # Balanced when no data
        )

        # Normalize: 0.5 is balanced, map 0.0-1.0 to 0.0-1.0
        # ratio > 0.5 means more down, ratio < 0.5 means more up
        # We only care about down-biased asymmetry, so max(0.5, ratio) gives us 0.5-1.0
        # Then we map 0.5-1.0 to 0.0-1.0
        normalized = (ratio - 0.5) * 2

        # Clamp to 0-1
        normalized = normalized.clip(0, 1)

        return pd.Series(normalized, index=down_metric.index)

    def get_current_state(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Get current speed asymmetry score and label.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (score, label)
        """
        if len(df) < self.lookback:
            return 0.0, "INSUFFICIENT_DATA"

        asymmetry = self.calculate(df)
        latest_score = asymmetry.iloc[-1]

        if latest_score < 0.3:
            label = "BALANCED"
        elif latest_score < 0.6:
            label = "TILTING_DOWN"
        elif latest_score < 0.8:
            label = "DOWNWARD_PRESSURE"
        else:
            label = "PANIC_SELLING"

        return latest_score, label


if __name__ == "__main__":
    # Example usage
    from data.fetch_binance import BinanceFetcher

    # Fetch some data
    fetcher = BinanceFetcher()
    df = fetcher.load_data('BTCUSDT', '1h')

    if df is not None:
        # Calculate speed asymmetry
        sa = SpeedAsymmetry()
        df['speed_asymmetry'] = sa.calculate(df)

        print(f"Latest speed asymmetry score: {df['speed_asymmetry'].iloc[-1]:.3f}")
        print(f"Average speed asymmetry: {df['speed_asymmetry'].mean():.3f}")

        # Get current state
        score, label = sa.get_current_state(df)
        print(f"\nCurrent state: {label} (score: {score:.3f})")
