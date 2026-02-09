"""
State Labeler

Maps continuous stress scores (0.0 → 1.0) to discrete labeled states
for easier interpretation and alerting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


class StateLabeler:
    """
    Maps stress scores to labeled market states.

    States:
    - NORMAL (0.0 - 0.3): Calm market conditions
    - STRESSED (0.3 - 0.6): Elevated stress, monitoring recommended
    - FRAGILE (0.6 - 0.8): Multiple stress conditions present, caution advised
    - IMMINENT_CLEARING (0.8 - 1.0): High probability of forced liquidation event
    """

    # State definitions (inclusive ranges)
    STATES = {
        'NORMAL': {'min': 0.0, 'max': 0.3, 'color': '🟢', 'description': 'Calm market'},
        'STRESSED': {'min': 0.3, 'max': 0.6, 'color': '🟡', 'description': 'Elevated stress'},
        'FRAGILE': {'min': 0.6, 'max': 0.8, 'color': '🟠', 'description': 'Fragile conditions'},
        'IMMINENT_CLEARING': {'min': 0.8, 'max': 1.0, 'color': '🔴', 'description': 'Imminent liquidation risk'}
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize state labeler.

        Args:
            thresholds: Custom threshold values (optional)
        """
        if thresholds:
            self._validate_thresholds(thresholds)
            self.thresholds = thresholds
        else:
            self.thresholds = {
                'normal_max': 0.3,
                'stressed_max': 0.6,
                'fragile_max': 0.8
            }

    def _validate_thresholds(self, thresholds: Dict[str, float]):
        """Validate custom thresholds are valid."""
        if not (0 < thresholds.get('normal_max', 0.3) <
                thresholds.get('stressed_max', 0.6) <
                thresholds.get('fragile_max', 0.8) <= 1.0):
            raise ValueError("Thresholds must be strictly increasing and ≤ 1.0")

    def get_state(self, stress_score: float) -> Tuple[str, Dict]:
        """
        Get labeled state for a given stress score.

        Args:
            stress_score: Stress score (0.0 → 1.0)

        Returns:
            Tuple of (state_label, state_metadata)
        """
        # Clamp score to valid range
        score = np.clip(stress_score, 0.0, 1.0)

        if score < self.thresholds['normal_max']:
            state = 'NORMAL'
        elif score < self.thresholds['stressed_max']:
            state = 'STRESSED'
        elif score < self.thresholds['fragile_max']:
            state = 'FRAGILE'
        else:
            state = 'IMMINENT_CLEARING'

        return state, self.STATES[state]

    def label_series(self, stress_scores: pd.Series) -> pd.Series:
        """
        Apply state labels to a series of stress scores.

        Args:
            stress_scores: Series of stress scores

        Returns:
            Series of state labels
        """
        return stress_scores.apply(lambda x: self.get_state(x)[0])

    def get_state_summary(
        self,
        df: pd.DataFrame,
        stress_score_col: str = 'stress_score'
    ) -> Dict:
        """
        Get summary statistics for each state.

        Args:
            df: DataFrame with stress scores
            stress_score_col: Name of stress score column

        Returns:
            Dictionary with state distribution and counts
        """
        if stress_score_col not in df.columns:
            raise ValueError(f"Column '{stress_score_col}' not found in DataFrame")

        labels = self.label_series(df[stress_score_col])

        summary = {}
        for state_name, state_info in self.STATES.items():
            count = (labels == state_name).sum()
            pct = (count / len(labels)) * 100 if len(labels) > 0 else 0
            summary[state_name] = {
                'count': count,
                'percentage': round(pct, 2),
                'emoji': state_info['color'],
                'description': state_info['description']
            }

        return summary

    def detect_state_transitions(
        self,
        df: pd.DataFrame,
        stress_score_col: str = 'stress_score'
    ) -> pd.DataFrame:
        """
        Detect when market state changes.

        Args:
            df: DataFrame with stress scores
            stress_score_col: Name of stress score column

        Returns:
            DataFrame with state transitions (timestamp, from_state, to_state, score)
        """
        if stress_score_col not in df.columns:
            raise ValueError(f"Column '{stress_score_col}' not found in DataFrame")

        df = df.copy()
        df['state'] = self.label_series(df[stress_score_col])

        # Find transitions
        df['state_changed'] = df['state'] != df['state'].shift()
        transitions = df[df['state_changed']].copy()

        if transitions.empty:
            return pd.DataFrame(columns=['timestamp', 'from_state', 'to_state', 'score'])

        transitions['from_state'] = transitions['state'].shift()
        transitions = transitions[transitions['from_state'].notna()]

        return transitions[['timestamp', 'from_state', 'to_state', stress_score_col]].copy()

    def format_state_alert(self, state: str, score: float) -> str:
        """
        Format a human-readable state alert.

        Args:
            state: State label
            score: Stress score

        Returns:
            Formatted alert string
        """
        _, state_info = self.get_state(score)
        emoji = state_info['color']
        description = state_info['description']

        return f"{emoji} {state} ({score:.3f}) - {description}"


if __name__ == "__main__":
    # Example usage
    from data.fetch_binance import BinanceFetcher
    from detector.stress_score import StressCalculator

    # Fetch some data
    fetcher = BinanceFetcher()
    df = fetcher.load_data('BTCUSDT', '1h')

    if df is not None:
        # Calculate stress score
        calculator = StressCalculator()
        df['stress_score'] = calculator.calculate(df)

        # Label states
        labeler = StateLabeler()
        df['state'] = labeler.label_series(df['stress_score'])

        # Get summary
        summary = labeler.get_state_summary(df)
        print("State distribution:")
        for state, info in summary.items():
            print(f"  {info['emoji']} {state}: {info['count']} candles ({info['percentage']}%)")

        # Get current state
        latest_score = df['stress_score'].iloc[-1]
        current_state, state_info = labeler.get_state(latest_score)
        print(f"\nCurrent state: {labeler.format_state_alert(current_state, latest_score)}")

        # Detect transitions
        transitions = labeler.detect_state_transitions(df)
        print(f"\nState transitions found: {len(transitions)}")
        if not transitions.empty:
            print(transitions.head(10))
