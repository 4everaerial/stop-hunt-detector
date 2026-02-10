"""
Rolling Normalized Stress Calculator

Applies rolling z-score normalization to each signal before combining.
This makes stress scores relative to recent history, reducing baseline inflation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from signals import (
    VolatilityCompression,
    LiquidityFragility,
    ContinuationFailure,
    SpeedAsymmetry
)


class RollingStressCalculator:
    """
    Calculates stress score with rolling z-score normalization.

    Each signal is normalized by its rolling mean and standard deviation
    over a lookback window (default: 168 hours = 7 days).

    Formula: z = (signal - rolling_mean) / rolling_std
    Then z is mapped to 0-1 using error function (erf).

    This makes stress scores relative to recent market conditions,
    reducing baseline inflation during naturally volatile periods.
    """

    # Signal weights
    DEFAULT_WEIGHTS = {
        'volatility': 0.35,
        'liquidity': 0.35,
        'continuation': 0.15,
        'speed': 0.15
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        lookback_hours: int = 168,  # 7 days
        **signal_params
    ):
        """
        Initialize rolling stress calculator.

        Args:
            weights: Signal weights (default: equal weighting)
            lookback_hours: Rolling window for z-score normalization (default: 168 hours)
            **signal_params: Parameters passed to individual signal detectors
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.lookback_hours = lookback_hours

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Signal weights must sum to 1.0, got {total_weight}")

        # Initialize signal detectors
        self.volatility = VolatilityCompression(**signal_params)
        self.liquidity = LiquidityFragility(**signal_params)
        self.continuation = ContinuationFailure(**signal_params)
        self.speed = SpeedAsymmetry(**signal_params)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling-normalized stress score.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of stress scores (0.0 → 1.0)
        """
        df = df.copy()

        # Calculate raw signals
        df['signal_volatility_raw'] = self.volatility.calculate(df)
        df['signal_liquidity_raw'] = self.liquidity.calculate(df)
        df['signal_continuation_raw'] = self.continuation.calculate(df)
        df['signal_speed_raw'] = self.speed.calculate(df)

        # Apply rolling z-score normalization
        df['signal_volatility'] = self._rolling_normalize(df['signal_volatility_raw'])
        df['signal_liquidity'] = self._rolling_normalize(df['signal_liquidity_raw'])
        df['signal_continuation'] = self._rolling_normalize(df['signal_continuation_raw'])
        df['signal_speed'] = self._rolling_normalize(df['signal_speed_raw'])

        # Calculate composite stress score
        df['stress_score'] = (
            df['signal_volatility'] * self.weights['volatility'] +
            df['signal_liquidity'] * self.weights['liquidity'] +
            df['signal_continuation'] * self.weights['continuation'] +
            df['signal_speed'] * self.weights['speed']
        )

        return df['stress_score']

    def _rolling_normalize(self, series: pd.Series) -> pd.Series:
        """
        Apply rolling z-score normalization.

        For each point, calculate z-score relative to rolling window:
        z = (x - mean) / std

        Then map z to 0-1 using error function:
        normalized = 0.5 * (1 + erf(z / sqrt(2)))

        This maps z-scores from [-inf, inf] to [0, 1],
        where z=0 → 0.5, z=-2 → ~0.02, z=2 → ~0.98

        Args:
            series: Signal series to normalize

        Returns:
            Normalized series (0.0 → 1.0)
        """
        from scipy.special import erf

        # Create a copy to avoid modifying original
        series_clean = series.copy()

        # For NaN values in the original series, we can't normalize them
        # We'll normalize only the valid values, then fill backward/forward

        # First, forward-fill raw NaN values with a reasonable default
        # This is needed for the rolling window to work
        series_clean = series_clean.ffill().bfill()

        # If still NaN (all NaN), return zeros
        if series_clean.isna().all():
            return pd.Series(0.5, index=series.index)

        # Calculate rolling mean and std
        rolling_mean = series_clean.rolling(window=self.lookback_hours, min_periods=1).mean()
        rolling_std = series_clean.rolling(window=self.lookback_hours, min_periods=1).std()

        # Handle NaN at start (use expanding window)
        rolling_mean = rolling_mean.ffill().fillna(series_clean.mean())
        rolling_std = rolling_std.ffill().fillna(series_clean.std())

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, series_clean.std())
        rolling_std = rolling_std.replace(0, 0.01)  # Minimum std to avoid division by zero

        # Calculate z-scores
        z_scores = (series_clean - rolling_mean) / rolling_std

        # Map z-scores to 0-1 using error function
        # erf(0) = 0, so we add 0.5 to center at 0.5
        normalized = 0.5 * (1 + erf(z_scores / np.sqrt(2)))

        # Now restore NaN values where the original series had NaN
        # This ensures we don't normalize invalid data
        normalized = normalized.where(~series.isna(), np.nan)

        return normalized

    def calculate_with_details(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate stress score and return detailed component breakdown.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (stress_score Series, details DataFrame)
        """
        df_working = df.copy()

        # Calculate raw signals
        df_working['signal_volatility_raw'] = self.volatility.calculate(df_working)
        df_working['signal_liquidity_raw'] = self.liquidity.calculate(df_working)
        df_working['signal_continuation_raw'] = self.continuation.calculate(df_working)
        df_working['signal_speed_raw'] = self.speed.calculate(df_working)

        # Apply rolling normalization
        df_working['signal_volatility'] = self._rolling_normalize(df_working['signal_volatility_raw'])
        df_working['signal_liquidity'] = self._rolling_normalize(df_working['signal_liquidity_raw'])
        df_working['signal_continuation'] = self._rolling_normalize(df_working['signal_continuation_raw'])
        df_working['signal_speed'] = self._rolling_normalize(df_working['signal_speed_raw'])

        # Calculate composite
        df_working['stress_score'] = (
            df_working['signal_volatility'] * self.weights['volatility'] +
            df_working['signal_liquidity'] * self.weights['liquidity'] +
            df_working['signal_continuation'] * self.weights['continuation'] +
            df_working['signal_speed'] * self.weights['speed']
        )

        details = df_working[[
            'signal_volatility',
            'signal_liquidity',
            'signal_continuation',
            'signal_speed',
            'stress_score'
        ]].copy()

        return df_working['stress_score'], details

    def get_current_stress(self, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Get current stress score and component breakdown.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (current stress score, component values)
        """
        stress_score, details = self.calculate_with_details(df)

        current_score = stress_score.iloc[-1]

        components = {
            'volatility': details['signal_volatility'].iloc[-1],
            'liquidity': details['signal_liquidity'].iloc[-1],
            'continuation': details['signal_continuation'].iloc[-1],
            'speed': details['signal_speed'].iloc[-1]
        }

        return current_score, components


if __name__ == "__main__":
    # Example usage
    from data.mock_generator_enhanced_fixed import generate_enhanced_stress_event_data

    # Generate test data
    df = generate_enhanced_stress_event_data(event_day=30, days=90)

    # Calculate rolling-normalized stress score
    calculator = RollingStressCalculator(lookback_hours=168)
    stress_score, details = calculator.calculate_with_details(df)

    print(f"Latest stress score: {stress_score.iloc[-1]:.3f}")
    print(f"Mean stress score: {stress_score.mean():.3f}")
    print(f"Std stress score: {stress_score.std():.3f}")
    print()

    current_score, components = calculator.get_current_stress(df)
    print("Component breakdown:")
    for name, value in components.items():
        print(f"  {name}: {value:.3f}")
