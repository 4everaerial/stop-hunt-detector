"""
Continuation Failure Signal

Detects when momentum indicators diverge from price action -
prices move but momentum doesn't follow, suggesting weakness.

Mechanism:
- RSI divergence: Price makes new high/low, RSI doesn't
- MACD divergence: Price vs MACD histogram
- Momentum deceleration: Rate of change slows

Score: 0.0 (strong continuation) → 1.0 (divergence/weakness)
"""

import pandas as pd
import numpy as np
from typing import Tuple


class ContinuationFailure:
    """
    Measures continuation failure / momentum divergence.

    High continuation failure indicates that price moves lack
    underlying momentum, often seen before reversals.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        lookback: int = 20
    ):
        """
        Initialize continuation failure detector.

        Args:
            rsi_period: RSI period (default: 14)
            macd_fast: MACD fast EMA period (default: 12)
            macd_slow: MACD slow EMA period (default: 26)
            macd_signal: MACD signal EMA period (default: 9)
            lookback: Lookback period for divergence detection (default: 20)
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate continuation failure score for each candle.

        Args:
            df: DataFrame with OHLCV data (columns: close)

        Returns:
            Series of continuation failure scores (0.0 → 1.0)
        """
        df = df.copy()

        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)

        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])

        # Calculate Rate of Change (momentum)
        df['roc'] = df['close'].pct_change(self.lookback)

        # Detect divergences
        df['rsi_divergence'] = self._detect_divergence(df['close'], df['rsi'])
        df['macd_divergence'] = self._detect_divergence(df['close'], df['macd_hist'])

        # Momentum deceleration: ROC magnitude vs price change magnitude
        df['price_change_abs'] = df['close'].diff().abs()
        df['roc_abs'] = df['roc'].abs()
        df['momentum_decel'] = self._normalize_momentum_deceleration(
            df['price_change_abs'],
            df['roc_abs']
        )

        # Composite score
        df['continuation_failure'] = (
            df['rsi_divergence'] * 0.4 +
            df['macd_divergence'] * 0.3 +
            df['momentum_decel'] * 0.3
        )

        return df['continuation_failure']

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.macd_signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _detect_divergence(self, price: pd.Series, indicator: pd.Series) -> pd.Series:
        """
        Detect divergence between price and indicator.

        Returns 1.0 if divergence detected, 0.0 otherwise.
        """
        # Simple divergence: price makes new high but indicator doesn't (bearish)
        # or price makes new low but indicator doesn't (bullish)
        lookback = 5

        price_high = price.rolling(lookback).max()
        price_low = price.rolling(lookback).min()

        ind_high = indicator.rolling(lookback).max()
        ind_low = indicator.rolling(lookback).min()

        # Bearish divergence: price at high, indicator not at high
        bearish = (price == price_high) & (indicator < ind_high)

        # Bullish divergence: price at low, indicator not at low
        bullish = (price == price_low) & (indicator > ind_low)

        divergence = bearish | bullish

        return divergence.astype(float)

    def _normalize_momentum_deceleration(
        self,
        price_change: pd.Series,
        roc: pd.Series
    ) -> pd.Series:
        """
        Normalize momentum deceleration.

        High price change with low ROC suggests momentum is fading.
        """
        # Normalize both to 0-1
        price_norm = self._rolling_normalize(price_change.rolling(self.lookback))
        roc_norm = self._rolling_normalize(roc.rolling(self.lookback))

        # Momentum deceleration: high price move but low ROC
        decel = price_norm * (1 - roc_norm)

        return decel

    def _rolling_normalize(self, rolling_obj) -> pd.Series:
        """Normalize rolling series to 0-1."""
        series = rolling_obj.min()
        rolling_max = rolling_obj.max()
        rolling_min = rolling_obj.min()

        normalized = np.where(
            rolling_max == rolling_min,
            0.5,
            (series - rolling_min) / (rolling_max - rolling_min)
        )

        return pd.Series(normalized, index=series.index)

    def get_current_state(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Get current continuation failure score and label.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (score, label)
        """
        if len(df) < self.lookback + self.macd_slow:
            return 0.0, "INSUFFICIENT_DATA"

        cf = self.calculate(df)
        latest_score = cf.iloc[-1]

        if latest_score < 0.3:
            label = "STRONG"
        elif latest_score < 0.6:
            label = "NORMAL"
        elif latest_score < 0.8:
            label = "WEAK"
        else:
            label = "FAILING"

        return latest_score, label
