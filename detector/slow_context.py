"""
Slow Context Layer

Long-horizon absolute market context using rolling percentile normalization.
Provides regime classification independent of fast relative stress.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from signals import (
    VolatilityCompression,
    LiquidityFragility,
    ContinuationFailure,
    SpeedAsymmetry
)


class SlowContextCalculator:
    """
    Calculates slow absolute context using long-horizon percentile normalization.

    Window: 180-365 days (non-stationary market adaptation)
    Method: Rolling percentile normalization to [0,1]
    Output: slow_context ∈ [0,1]

    Interpretation:
    - < 0.35 → Cold / Complacent regime
    - 0.35-0.65 → Neutral regime
    - > 0.65 → Hot / Stressed regime

    These are regime LABELS for interpretation, not execution triggers.
    """

    # Same weights as fast detector for consistency
    DEFAULT_WEIGHTS = {
        'volatility': 0.35,
        'liquidity': 0.35,
        'continuation': 0.15,
        'speed': 0.15
    }

    def __init__(
        self,
        weights: Dict[str, float] = None,
        lookback_hours: int = 4380,  # 180 days (non-stationary markets)
        **signal_params
    ):
        """
        Initialize slow context calculator.

        Args:
            weights: Signal weights (default: same as fast detector)
            lookback_hours: Rolling window for percentile normalization (default: 180 days = 4380 hours)
            **signal_params: Parameters passed to individual signal detectors
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Signal weights must sum to 1.0, got {total_weight}")

        self.lookback_hours = lookback_hours

        # Initialize signal detectors (same as fast detector)
        self.volatility = VolatilityCompression(**signal_params)
        self.liquidity = LiquidityFragility(**signal_params)
        self.continuation = ContinuationFailure(**signal_params)
        self.speed = SpeedAsymmetry(**signal_params)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate slow context score.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of slow context scores (0.0 → 1.0)
        """
        df = df.copy()

        # Calculate raw signals (same as fast detector)
        df['signal_volatility_raw'] = self.volatility.calculate(df)
        df['signal_liquidity_raw'] = self.liquidity.calculate(df)
        df['signal_continuation_raw'] = self.continuation.calculate(df)
        df['signal_speed_raw'] = self.speed.calculate(df)

        # Calculate composite raw stress (before any normalization)
        df['raw_stress'] = (
            df['signal_volatility_raw'] * self.weights['volatility'] +
            df['signal_liquidity_raw'] * self.weights['liquidity'] +
            df['signal_continuation_raw'] * self.weights['continuation'] +
            df['signal_speed_raw'] * self.weights['speed']
        )

        # Apply rolling percentile normalization
        df['slow_context'] = self._rolling_percentile_normalize(df['raw_stress'])

        return df['slow_context']

    def _rolling_percentile_normalize(self, series: pd.Series) -> pd.Series:
        """
        Apply rolling percentile normalization to [0,1].

        For each point, calculate percentile rank within rolling window:
        percentile = rank / window_size

        Maps to 0-1 where:
        - 0.0 = lowest stress in window
        - 0.5 = median stress in window
        - 1.0 = highest stress in window

        Args:
            series: Signal series to normalize

        Returns:
            Normalized series (0.0 → 1.0)
        """
        # Calculate rolling rank (percentile)
        # Use expanding window at start, then rolling
        window_size = min(self.lookback_hours, len(series))

        # Calculate rank for each point relative to rolling window
        # This gives percentile from 0 to 1
        normalized = np.zeros(len(series))

        for i in range(len(series)):
            # Determine window
            start_idx = max(0, i - self.lookback_hours)
            window = series.iloc[start_idx:i+1]

            if len(window) == 0:
                normalized[i] = 0.5
                continue

            # Calculate percentile rank
            # Count values <= current value, divide by window size
            rank = (window <= series.iloc[i]).sum()
            percentile = rank / len(window)

            # Clamp to [0,1] (handle edge cases)
            percentile = np.clip(percentile, 0.0, 1.0)
            normalized[i] = percentile

        return pd.Series(normalized, index=series.index)

    def get_regime_label(self, slow_context: float) -> Tuple[str, str]:
        """
        Get regime label for a given slow context score.

        Args:
            slow_context: Slow context score (0.0 → 1.0)

        Returns:
            Tuple of (regime_label, description)
        """
        if slow_context < 0.35:
            return 'COLD', 'Complacent regime - historically low stress'
        elif slow_context < 0.65:
            return 'NEUTRAL', 'Neutral regime - average stress'
        else:
            return 'HOT', 'Stressed regime - historically high stress'

    def calculate_with_details(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate slow context and return detailed breakdown.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (slow_context Series, details DataFrame)
        """
        df_working = df.copy()

        # Calculate raw signals
        df_working['signal_volatility_raw'] = self.volatility.calculate(df_working)
        df_working['signal_liquidity_raw'] = self.liquidity.calculate(df_working)
        df_working['signal_continuation_raw'] = self.continuation.calculate(df_working)
        df_working['signal_speed_raw'] = self.speed.calculate(df_working)

        # Calculate composite raw stress
        df_working['raw_stress'] = (
            df_working['signal_volatility_raw'] * self.weights['volatility'] +
            df_working['signal_liquidity_raw'] * self.weights['liquidity'] +
            df_working['signal_continuation_raw'] * self.weights['continuation'] +
            df_working['signal_speed_raw'] * self.weights['speed']
        )

        # Apply rolling percentile normalization
        df_working['slow_context'] = self._rolling_percentile_normalize(df_working['raw_stress'])

        # Calculate regime labels
        df_working['regime'], df_working['regime_description'] = zip(
            *df_working['slow_context'].apply(self.get_regime_label)
        )

        details = df_working[[
            'signal_volatility_raw',
            'signal_liquidity_raw',
            'signal_continuation_raw',
            'signal_speed_raw',
            'raw_stress',
            'slow_context',
            'regime'
        ]].copy()

        return df_working['slow_context'], details

    def get_current_context(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Get current slow context and regime label.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (slow_context_score, regime_label)
        """
        slow_context = self.calculate(df)
        latest_score = slow_context.iloc[-1]
        regime, description = self.get_regime_label(latest_score)

        return latest_score, regime


if __name__ == "__main__":
    # Example usage
    from data.mock_generator_enhanced_fixed import generate_enhanced_stress_event_data

    # Generate test data
    df = generate_enhanced_stress_event_data(event_day=30, days=90)
    print(f"Generated {len(df)} candles\n")

    # Calculate slow context
    calculator = SlowContextCalculator(lookback_hours=4380)
    slow_context, details = calculator.calculate_with_details(df)

    print(f"Latest slow context: {slow_context.iloc[-1]:.3f}")
    print(f"Mean slow context: {slow_context.mean():.3f}")
    print(f"Std slow context: {slow_context.std():.3f}")
    print()

    latest_score, regime = calculator.get_current_context(df)
    print("Current regime:")
    print(f"  Slow context: {latest_score:.3f}")
    print(f"  Regime: {regime}")
