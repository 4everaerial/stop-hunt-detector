"""
Volatility Compression Signal

Detects when trading range tightens and volatility decays - a classic
precursor to explosive price moves and stop hunts.

Mechanism:
- Bollinger Band squeeze: width narrows significantly
- ATR decay: True Range decreases over time
- Range contraction: High-Low range shrinks relative to price

Score: 0.0 (normal) → 1.0 (severely compressed)
"""

import pandas as pd
import numpy as np
from typing import Tuple


class VolatilityCompression:
    """
    Measures volatility compression as a precursor to stop hunts.

    High volatility compression scores indicate that the market is
    "coiling up" and may soon explode in either direction.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        lookback: int = 100
    ):
        """
        Initialize volatility compression detector.

        Args:
            bb_period: Bollinger Band period (default: 20)
            bb_std: Bollinger Band standard deviations (default: 2.0)
            atr_period: ATR period (default: 14)
            lookback: Lookback period for compression calculation (default: 100)
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility compression score for each candle.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close)

        Returns:
            Series of compression scores (0.0 → 1.0)
        """
        df = df.copy()

        # Calculate True Range (ATR component)
        df['tr'] = self._true_range(df)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(self.bb_period).mean()
        df['std'] = df['close'].rolling(self.bb_period).std()
        df['upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['lower'] = df['sma'] - (df['std'] * self.bb_std)
        df['bb_width'] = (df['upper'] - df['lower']) / df['sma']

        # Calculate High-Low range as % of price
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        # Normalized signals (0 = wide/volatile, 1 = compressed)
        df['bb_squeeze'] = self._normalize_inverse(df['bb_width'].rolling(self.lookback))
        df['atr_decay'] = self._normalize_inverse(df['atr'].rolling(self.lookback))
        df['range_contraction'] = self._normalize_inverse(df['hl_range'].rolling(self.lookback))

        # Composite score (equal weight for now)
        df['compression'] = (
            df['bb_squeeze'] * 0.4 +
            df['atr_decay'] * 0.3 +
            df['range_contraction'] * 0.3
        )

        return df['compression']

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range (max of high-low, high-prev_close, low-prev_close)."""
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _normalize_inverse(self, rolling_obj) -> pd.Series:
        """
        Normalize rolling metric inverse (low values → high score).

        Low volatility/width should give high compression score.
        """
        series = rolling_obj.min()  # Get minimum over window

        # Inverse normalization: current / max gives 0-1, but we want inverse
        # So we use: max - current / (max - min) with clamping
        rolling_max = rolling_obj.max()
        rolling_min = rolling_obj.min()

        # Handle edge case where max = min
        normalized = np.where(
            rolling_max == rolling_min,
            0.5,  # Middle value when no variation
            (rolling_max - series) / (rolling_max - rolling_min)
        )

        return pd.Series(normalized, index=series.index)

    def get_current_state(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Get current compression score and label.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (score, label)
        """
        if len(df) < self.lookback:
            return 0.0, "INSUFFICIENT_DATA"

        compression = self.calculate(df)
        latest_score = compression.iloc[-1]

        if latest_score < 0.3:
            label = "NORMAL"
        elif latest_score < 0.6:
            label = "COMPRESSING"
        elif latest_score < 0.8:
            label = "TIGHT"
        else:
            label = "COILED"

        return latest_score, label


if __name__ == "__main__":
    # Example usage
    from data.fetch_binance import BinanceFetcher

    # Fetch some data
    fetcher = BinanceFetcher()
    df = fetcher.load_data('BTCUSDT', '1h')

    if df is not None:
        # Calculate compression
        vc = VolatilityCompression()
        df['compression'] = vc.calculate(df)

        print(f"Latest compression score: {df['compression'].iloc[-1]:.3f}")
        print(f"Average compression: {df['compression'].mean():.3f}")

        # Get current state
        score, label = vc.get_current_state(df)
        print(f"\nCurrent state: {label} (score: {score:.3f})")
