"""
Composite Stress Score Calculator

Combines individual stress signals into a unified stress score (0.0 → 1.0).
This is the core output of the detector system.
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


class StressCalculator:
    """
    Calculates a composite market stress score from individual signals.

    The stress score (0.0 → 1.0) represents the probability of imminent
    forced-liquidation / stop-hunt events based on market conditions.
    """

    # Signal weights (empirically tuned via weight_tuner.py)
    # Top performer on simulated stress events:
    # volatility: 35%, liquidity: 35%, continuation: 15%, speed: 15%
    DEFAULT_WEIGHTS = {
        'volatility': 0.35,     # Volatility compression
        'liquidity': 0.35,      # Liquidity fragility
        'continuation': 0.15,   # Continuation failure
        'speed': 0.15           # Speed asymmetry
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        **signal_params
    ):
        """
        Initialize stress calculator.

        Args:
            weights: Signal weights (default: equal weighting)
            **signal_params: Parameters passed to individual signal detectors
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

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
        Calculate composite stress score for each candle.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            Series of stress scores (0.0 → 1.0)
        """
        df = df.copy()

        # Calculate individual signals
        df['signal_volatility'] = self.volatility.calculate(df)
        df['signal_liquidity'] = self.liquidity.calculate(df)
        df['signal_continuation'] = self.continuation.calculate(df)
        df['signal_speed'] = self.speed.calculate(df)

        # Calculate composite stress score
        df['stress_score'] = (
            df['signal_volatility'] * self.weights['volatility'] +
            df['signal_liquidity'] * self.weights['liquidity'] +
            df['signal_continuation'] * self.weights['continuation'] +
            df['signal_speed'] * self.weights['speed']
        )

        return df['stress_score']

    def calculate_with_details(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate stress score and return detailed component breakdown.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (stress_score Series, details DataFrame with all components)
        """
        # Re-calculate signals to get df with all columns
        df_working = df.copy()
        df_working['signal_volatility'] = self.volatility.calculate(df_working)
        df_working['signal_liquidity'] = self.liquidity.calculate(df_working)
        df_working['signal_continuation'] = self.continuation.calculate(df_working)
        df_working['signal_speed'] = self.speed.calculate(df_working)
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
            Tuple of (current stress score, component values dict)
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

    def find_high_stress_periods(
        self,
        df: pd.DataFrame,
        threshold: float = 0.7,
        min_duration: int = 1
    ) -> pd.DataFrame:
        """
        Find periods where stress exceeded threshold.

        Args:
            df: DataFrame with OHLCV data
            threshold: Stress score threshold (default: 0.7)
            min_duration: Minimum candle count (default: 1)

        Returns:
            DataFrame with high-stress periods (start, end, duration, peak_stress)
        """
        df = df.copy()
        df['stress_score'] = self.calculate(df)

        # Find candles above threshold
        df['above_threshold'] = df['stress_score'] >= threshold

        # Find contiguous periods
        df['period_id'] = (df['above_threshold'] != df['above_threshold'].shift()).cumsum()
        df['above_threshold_shifted'] = df['above_threshold'].shift()

        # Filter to periods above threshold
        high_stress = df[df['above_threshold']].copy()

        if high_stress.empty:
            return pd.DataFrame(columns=['start', 'end', 'duration', 'peak_stress'])

        # Group by period
        periods = high_stress.groupby('period_id').agg({
            'timestamp': ['min', 'max'],
            'stress_score': 'max',
            'period_id': 'count'
        }).reset_index()

        periods.columns = ['period_id', 'start', 'end', 'peak_stress', 'duration']

        # Filter by minimum duration
        periods = periods[periods['duration'] >= min_duration]

        return periods[['start', 'end', 'duration', 'peak_stress']]


if __name__ == "__main__":
    # Example usage
    from data.fetch_binance import BinanceFetcher

    # Fetch some data
    fetcher = BinanceFetcher()
    df = fetcher.load_data('BTCUSDT', '1h')

    if df is not None:
        # Calculate stress score
        calculator = StressCalculator()
        stress_score, details = calculator.calculate_with_details(df)

        print(f"Latest stress score: {stress_score.iloc[-1]:.3f}")
        print(f"\nComponent breakdown:")
        current_score, components = calculator.get_current_stress(df)
        for name, value in components.items():
            print(f"  {name}: {value:.3f} (weight: {calculator.weights[name]})")

        # Find high-stress periods
        high_stress = calculator.find_high_stress_periods(df, threshold=0.7)
        print(f"\nHigh-stress periods found: {len(high_stress)}")
        if not high_stress.empty:
            print(high_stress.head())
