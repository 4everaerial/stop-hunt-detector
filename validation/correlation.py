"""
Correlation Analysis

Calculates correlation metrics between stress scores and liquidation events.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


class CorrelationAnalyzer:
    """
    Analyzes correlation between stress scores and liquidation events.

    Metrics:
    - Pearson correlation: Linear relationship between stress and events
    - Spearman correlation: Monotonic relationship
    - Event binary correlation: Correlation with binary event occurrence
    - Lead-lag analysis: Correlation at different time lags
    """

    def __init__(self):
        """Initialize correlation analyzer."""
        pass

    def calculate_event_correlation(
        self,
        df: pd.DataFrame,
        events: List[Dict],
        stress_col: str = 'stress_score',
        lookback_hours: int = 4
    ) -> Dict[str, float]:
        """
        Calculate correlation between stress scores and event occurrence.

        Creates a binary event indicator and correlates with stress.

        Args:
            df: DataFrame with stress scores and timestamps
            events: List of event dicts with 'timestamp'
            stress_col: Name of stress score column
            lookback_hours: Hours before event to consider as event period

        Returns:
            Dictionary with correlation metrics
        """
        df = df.copy()

        # Create binary event indicator (1 = event occurred within lookback window)
        event_times = [pd.to_datetime(e['timestamp']) for e in events]
        df['event_occurred'] = 0

        for event_time in event_times:
            # Mark candles within lookback window as "event period"
            mask = (df['timestamp'] >= event_time - pd.Timedelta(hours=lookback_hours)) & \
                   (df['timestamp'] <= event_time + pd.Timedelta(hours=1))
            df.loc[mask, 'event_occurred'] = 1

        # Calculate correlations
        pearson_corr, pearson_p = stats.pearsonr(df[stress_col], df['event_occurred'])
        spearman_corr, spearman_p = stats.spearmanr(df[stress_col], df['event_occurred'])

        # Point-biserial correlation (special case of Pearson for binary variable)
        # Equivalent to pearson_corr, but calculated differently for clarity
        point_biserial = pearson_corr

        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'point_biserial': point_biserial
        }

    def lead_lag_correlation(
        self,
        df: pd.DataFrame,
        events: List[Dict],
        stress_col: str = 'stress_score',
        max_lag_hours: int = 24,
        lag_interval_hours: int = 1
    ) -> pd.DataFrame:
        """
        Calculate correlation at different time lags.

        Tests if stress scores lead events (positive lag) or follow events (negative lag).

        Args:
            df: DataFrame with stress scores and timestamps
            events: List of event dicts with 'timestamp'
            stress_col: Name of stress score column
            max_lag_hours: Maximum lag to test (default: 24 hours)
            lag_interval_hours: Lag interval (default: 1 hour)

        Returns:
            DataFrame with lag, correlation, and p-value
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Create binary event indicator
        event_times = [pd.to_datetime(e['timestamp']) for e in events]
        df['event_occurred'] = 0

        for event_time in event_times:
            # Event window: 1 hour after event time
            mask = (df['timestamp'] >= event_time) & \
                   (df['timestamp'] < event_time + pd.Timedelta(hours=1))
            df.loc[mask, 'event_occurred'] = 1

        results = []

        # Test different lags
        for lag_hours in range(-max_lag_hours, max_lag_hours + 1, lag_interval_hours):
            # Shift stress score by lag (positive = stress leads event)
            lag_candles = lag_hours  # Assuming 1h candles
            df['stress_lagged'] = df[stress_col].shift(-lag_candles)

            # Drop NaNs from shifting
            valid_df = df[df['stress_lagged'].notna() & (df['event_occurred'].notna())]

            if len(valid_df) < 10:
                continue

            # Calculate correlation
            corr, p_value = stats.pearsonr(valid_df['stress_lagged'], valid_df['event_occurred'])

            results.append({
                'lag_hours': lag_hours,
                'lag_candles': lag_candles,
                'correlation': corr,
                'p_value': p_value,
                'sample_size': len(valid_df)
            })

        return pd.DataFrame(results)

    def calculate_stress_elevation_metrics(
        self,
        df: pd.DataFrame,
        events: List[Dict],
        stress_col: str = 'stress_score',
        before_hours: int = 4,
        after_hours: int = 2,
        baseline_hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate stress elevation metrics around events.

        Compares stress before events vs baseline.

        Args:
            df: DataFrame with stress scores and timestamps
            events: List of event dicts with 'timestamp'
            stress_col: Name of stress score column
            before_hours: Hours before event to measure
            after_hours: Hours after event to measure
            baseline_hours: Hours before "before" window for baseline

        Returns:
            Dictionary with stress elevation metrics
        """
        results = {
            'pre_event_stress': [],
            'event_stress': [],
            'post_event_stress': [],
            'baseline_stress': []
        }

        for event in events:
            event_time = pd.to_datetime(event['timestamp'])

            # Pre-event window
            pre_start = event_time - pd.Timedelta(hours=before_hours)
            pre_end = event_time
            pre_mask = (df['timestamp'] >= pre_start) & (df['timestamp'] < pre_end)
            pre_stress = df.loc[pre_mask, stress_col].values

            # Event window (including after_hours)
            event_end = event_time + pd.Timedelta(hours=after_hours)
            event_mask = (df['timestamp'] >= event_time) & (df['timestamp'] < event_end)
            event_stress = df.loc[event_mask, stress_col].values

            # Post-event window
            post_start = event_end
            post_end = event_end + pd.Timedelta(hours=after_hours)
            post_mask = (df['timestamp'] >= post_start) & (df['timestamp'] < post_end)
            post_stress = df.loc[post_mask, stress_col].values

            # Baseline window
            baseline_start = pre_start - pd.Timedelta(hours=baseline_hours)
            baseline_end = pre_start
            baseline_mask = (df['timestamp'] >= baseline_start) & (df['timestamp'] < baseline_end)
            baseline_stress = df.loc[baseline_mask, stress_col].values

            if len(pre_stress) > 0:
                results['pre_event_stress'].append(np.mean(pre_stress))
            if len(event_stress) > 0:
                results['event_stress'].append(np.mean(event_stress))
            if len(post_stress) > 0:
                results['post_event_stress'].append(np.mean(post_stress))
            if len(baseline_stress) > 0:
                results['baseline_stress'].append(np.mean(baseline_stress))

        # Calculate aggregates
        metrics = {
            'avg_pre_event_stress': np.mean(results['pre_event_stress']) if results['pre_event_stress'] else None,
            'avg_event_stress': np.mean(results['event_stress']) if results['event_stress'] else None,
            'avg_post_event_stress': np.mean(results['post_event_stress']) if results['post_event_stress'] else None,
            'avg_baseline_stress': np.mean(results['baseline_stress']) if results['baseline_stress'] else None,
            'stress_elevation': None,
            'stress_elevation_ratio': None
        }

        # Calculate elevation
        if results['pre_event_stress'] and results['baseline_stress']:
            metrics['stress_elevation'] = metrics['avg_pre_event_stress'] - metrics['avg_baseline_stress']
            if metrics['avg_baseline_stress'] > 0:
                metrics['stress_elevation_ratio'] = metrics['avg_pre_event_stress'] / metrics['avg_baseline_stress']

        return metrics

    def calculate_signal_to_noise(
        self,
        df: pd.DataFrame,
        stress_col: str = 'stress_score',
        event_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Calculate signal-to-noise ratio.

        Compares stress during high-stress periods vs baseline.

        Args:
            df: DataFrame with stress scores
            stress_col: Name of stress score column
            event_threshold: Stress threshold for "signal" periods

        Returns:
            Dictionary with signal-to-noise metrics
        """
        # Signal: periods with high stress
        signal_mask = df[stress_col] >= event_threshold
        signal_stress = df.loc[signal_mask, stress_col]

        # Noise: baseline periods
        noise_mask = df[stress_col] < event_threshold
        noise_stress = df.loc[noise_mask, stress_col]

        metrics = {
            'signal_mean': signal_stress.mean() if len(signal_stress) > 0 else None,
            'signal_std': signal_stress.std() if len(signal_stress) > 0 else None,
            'noise_mean': noise_stress.mean() if len(noise_stress) > 0 else None,
            'noise_std': noise_stress.std() if len(noise_stress) > 0 else None,
            'signal_to_noise_ratio': None
        }

        if metrics['noise_mean'] and metrics['noise_std'] > 0:
            # Signal-to-noise as (signal_mean - noise_mean) / noise_std
            if metrics['signal_mean']:
                metrics['signal_to_noise_ratio'] = (metrics['signal_mean'] - metrics['noise_mean']) / metrics['noise_std']

        return metrics


if __name__ == "__main__":
    # Example usage
    from data.fetch_binance import BinanceFetcher
    from detector.stress_score import StressCalculator
    from data.tag_events import EventTagger

    # Fetch data
    fetcher = BinanceFetcher()
    df = fetcher.load_data('BTCUSDT', '1h')

    if df is not None:
        # Calculate stress scores
        calculator = StressCalculator()
        df['stress_score'] = calculator.calculate(df)

        # Load events
        tagger = EventTagger()
        events = tagger.list_events()

        # Run correlation analysis
        analyzer = CorrelationAnalyzer()

        # Event correlation
        corr = analyzer.calculate_event_correlation(df, events)
        print("Event Correlation:")
        print(f"  Pearson: {corr['pearson_correlation']:.3f} (p={corr['pearson_p_value']:.4f})")
        print(f"  Spearman: {corr['spearman_correlation']:.3f} (p={corr['spearman_p_value']:.4f})")

        # Stress elevation
        elevation = analyzer.calculate_stress_elevation_metrics(df, events)
        print("\nStress Elevation:")
        print(f"  Baseline: {elevation['avg_baseline_stress']:.3f}")
        print(f"  Pre-event: {elevation['avg_pre_event_stress']:.3f}")
        print(f"  Elevation: {elevation['stress_elevation']:.3f}")
        print(f"  Ratio: {elevation['stress_elevation_ratio']:.2f}x")

        # Signal-to-noise
        sn = analyzer.calculate_signal_to_noise(df)
        print("\nSignal-to-Noise:")
        print(f"  Signal mean: {sn['signal_mean']:.3f}")
        print(f"  Noise mean: {sn['noise_mean']:.3f}")
        print(f"  S/N ratio: {sn['signal_to_noise_ratio']:.3f}")
