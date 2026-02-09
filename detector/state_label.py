"""
State labeler for stress scores.

Converts continuous stress score (0-1) to labeled states:
- NORMAL (0.0-0.3)
- STRESSED (0.3-0.6)
- FRAGILE (0.6-0.8)
- IMMINENT_CLEARING (0.8-1.0)
"""

import pandas as pd


class StateLabeler:
    """Convert stress scores to labeled states."""

    def __init__(self, thresholds=None):
        """
        Initialize state labeler.

        Args:
            thresholds: Dictionary of state thresholds (default: 0.3, 0.6, 0.8)
                       e.g., {'normal_max': 0.3, 'stressed_max': 0.6, ...}
        """
        if thresholds is None:
            self.thresholds = {
                'normal_max': 0.3,
                'stressed_max': 0.6,
                'fragile_max': 0.8
            }
        else:
            self.thresholds = thresholds

    def label_state(self, df, score_col='stress_score'):
        """
        Add state labels based on stress score.

        Returns DataFrame with:
        - state: Labeled state (NORMAL/STRESSED/FRAGILE/IMMINENT_CLEARING)

        Args:
            df: DataFrame with stress scores
            score_col: Column name for stress score

        Returns:
            DataFrame with state labels
        """
        df = df.copy()

        # Define labeling function
        def get_state(score):
            if pd.isna(score):
                return 'UNKNOWN'
            elif score <= self.thresholds['normal_max']:
                return 'NORMAL'
            elif score <= self.thresholds['stressed_max']:
                return 'STRESSED'
            elif score <= self.thresholds['fragile_max']:
                return 'FRAGILE'
            else:
                return 'IMMINENT_CLEARING'

        # Apply labeling
        df['state'] = df[score_col].apply(get_state)

        return df

    def update_thresholds(self, new_thresholds):
        """
        Update state thresholds.

        Args:
            new_thresholds: Dictionary of new thresholds
        """
        self.thresholds.update(new_thresholds)
        print(f"Updated thresholds: {self.thresholds}")


def label_state(df, thresholds=None, score_col='stress_score'):
    """
    Add state labels (functional API).

    Args:
        df: DataFrame with stress scores
        thresholds: Dictionary of state thresholds
        score_col: Column name for stress score

    Returns:
        DataFrame with state labels
    """
    labeler = StateLabeler(thresholds)
    return labeler.label_state(df, score_col)
