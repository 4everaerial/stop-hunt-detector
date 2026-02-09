"""
Liquidity Fragility Signal

Detects when order book liquidity thins out and large wicks appear
on relatively low volume - signs of fragile liquidity.

Mechanism:
- Wick-to-body ratio: Large wicks relative to candle body
- Volume-to-range ratio: Low volume for large price moves
- Rejection patterns: Quick reversals from extremes

Score: 0.0 (deep liquidity) → 1.0 (fragile/illiquid)
"""

import pandas as pd
import numpy as np
from typing import Tuple


class LiquidityFragility:
    """
    Measures liquidity fragility in the market.

    High liquidity fragility indicates thin order books and
    easy manipulation, often seen before stop hunts.
    """

    def __init__(
        self,
        lookback: int = 20,
        wick_threshold: float = 0.5
    ):
        """
        Initialize liquidity fragility detector.

        Args:
            lookback: Lookback period for normalization (default: 20)
            wick_threshold: Threshold for "large wick" as % of range (default: 0.5)
        """
        self.lookback = lookback
        self.wick_threshold = wick_threshold

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity fragility score for each candle.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            Series of fragility scores (0.0 → 1.0)
        """
        df = df.copy()

        # Calculate wick ratios
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body'] = (df['close'] - df['open']).abs()
        df['total_range'] = df['high'] - df['low']

        # Wick-to-body ratio (handle small bodies)
        df['wick_body_ratio'] = np.where(
            df['body'] > 0,
            (df['upper_wick'] + df['lower_wick']) / df['body'],
            (df['upper_wick'] + df['lower_wick']) / df['total_range']
        )

        # Volume-to-range ratio: low volume + large range = fragility
        df['vol_range_ratio'] = df['volume'] / df['total_range']

        # Reaction magnitude: how much price rejected from extremes
        df['rejection'] = (df['upper_wick'] + df['lower_wick']) / df['total_range']

        # Normalize signals (0 = healthy, 1 = fragile)
        df['fragility_wick'] = self._normalize_wick_ratio(df['wick_body_ratio'])
        df['fragility_volume'] = self._normalize_inverse(df['vol_range_ratio'].rolling(self.lookback))
        df['fragility_rejection'] = df['rejection']  # Already 0-1

        # Composite score
        df['fragility'] = (
            df['fragility_wick'] * 0.4 +
            df['fragility_volume'] * 0.4 +
            df['fragility_rejection'] * 0.2
        )

        return df['fragility']

    def _normalize_wick_ratio(self, series: pd.Series) -> pd.Series:
        """Normalize wick-to-body ratio (high ratio → high fragility)."""
        # Cap extreme values
        capped = series.clip(upper=10)

        # Normalize 0-1
        return capped / 10.0

    def _normalize_inverse(self, rolling_obj) -> pd.Series:
        """Normalize rolling metric inverse (low values → high score)."""
        series = rolling_obj.min()
        rolling_max = rolling_obj.max()
        rolling_min = rolling_obj.min()

        normalized = np.where(
            rolling_max == rolling_min,
            0.5,
            (rolling_max - series) / (rolling_max - rolling_min)
        )

        return pd.Series(normalized, index=series.index)

    def get_current_state(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Get current fragility score and label.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (score, label)
        """
        if len(df) < self.lookback:
            return 0.0, "INSUFFICIENT_DATA"

        fragility = self.calculate(df)
        latest_score = fragility.iloc[-1]

        if latest_score < 0.3:
            label = "DEEP"
        elif latest_score < 0.6:
            label = "NORMAL"
        elif latest_score < 0.8:
            label = "THINNING"
        else:
            label = "FRAGILE"

        return latest_score, label
