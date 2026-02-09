"""
Composite stress score calculation.

Combines all stress signals into a single stress score (0.0→1.0).
"""

import pandas as pd
import numpy as np


class StressScoreDetector:
    """Calculate composite market stress score from individual signals."""

    def __init__(self, weights=None):
        """
        Initialize stress score detector.

        Args:
            weights: Dictionary of signal weights (default: equal weights)
                     e.g., {'volatility': 0.25, 'liquidity': 0.25, ...}
        """
        if weights is None:
            # Equal weights by default
            self.weights = {
                'volatility': 0.25,
                'liquidity': 0.25,
                'continuation': 0.25,
                'speed': 0.25
            }
        else:
            self.weights = weights

    def calculate_stress_score(self, df):
        """
        Calculate composite stress score from signal columns.

        Expects DataFrame with columns:
        - volatility_score (0-1)
        - liquidity_score (0-1)
        - continuation_score (0-1)
        - speed_score (0-1)

        Returns DataFrame with:
        - stress_score: Composite score (0-1)
        - signal_breakdown: Individual signal contributions

        Args:
            df: DataFrame with individual signal scores

        Returns:
            DataFrame with composite stress score
        """
        df = df.copy()

        # Check if all required columns exist
        required_cols = ['volatility_score', 'liquidity_score',
                        'continuation_score', 'speed_score']

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            # If signals not calculated yet, return zeros
            for col in required_cols:
                df[col] = 0.0
            print(f"Warning: Missing signal columns: {missing_cols}. Setting to 0.")

        # Calculate weighted composite score
        df['stress_score'] = (
            df['volatility_score'].fillna(0) * self.weights['volatility'] +
            df['liquidity_score'].fillna(0) * self.weights['liquidity'] +
            df['continuation_score'].fillna(0) * self.weights['continuation'] +
            df['speed_score'].fillna(0) * self.weights['speed']
        )

        # Clip to 0-1 range
        df['stress_score'] = df['stress_score'].clip(0, 1)

        # Add signal breakdown for debugging
        df['signal_breakdown'] = df.apply(
            lambda row: f"V:{row['volatility_score']:.2f}|L:{row['liquidity_score']:.2f}|C:{row['continuation_score']:.2f}|S:{row['speed_score']:.2f}",
            axis=1
        )

        return df

    def update_weights(self, new_weights):
        """
        Update signal weights.

        Args:
            new_weights: Dictionary of new weights
        """
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total:.3f}, normalizing to 1.0")
            for key in new_weights:
                new_weights[key] = new_weights[key] / total

        self.weights = new_weights
        print(f"Updated weights: {self.weights}")


def calculate_stress_score(df, weights=None):
    """
    Calculate composite stress score (functional API).

    Args:
        df: DataFrame with individual signal scores
        weights: Dictionary of signal weights

    Returns:
        DataFrame with composite stress score
    """
    detector = StressScoreDetector(weights)
    return detector.calculate_stress_score(df)
