"""
Historical Backtester

Validates stress scores against historical liquidation events.
Measures stress elevation before known events and generates reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
import json

from detector.stress_score import StressCalculator
from detector.state_label import StateLabeler


class Backtester:
    """
    Backtests stress scores against historical liquidation events.

    Metrics tracked:
    - Stress elevation before events (1-4 hours before)
    - Stress peak at event time
    - State transitions before/during events
    - Signal-to-noise ratio (baseline stress vs pre-event)
    """

    def __init__(
        self,
        stress_calculator: Optional[StressCalculator] = None,
        state_labeler: Optional[StateLabeler] = None
    ):
        """
        Initialize backtester.

        Args:
            stress_calculator: StressCalculator instance
            state_labeler: StateLabeler instance
        """
        self.stress_calculator = stress_calculator or StressCalculator()
        self.state_labeler = state_labeler or StateLabeler()

    def run_validation(
        self,
        df: pd.DataFrame,
        events: List[Dict],
        lookback_hours: int = 4,
        lookahead_hours: int = 2
    ) -> Dict:
        """
        Run validation against a list of liquidation events.

        Args:
            df: DataFrame with OHLCV data
            events: List of event dicts with 'timestamp' and 'pair'
            lookback_hours: Hours before event to check stress (default: 4)
            lookahead_hours: Hours after event to check stress (default: 2)

        Returns:
            Dictionary with validation results and metrics
        """
        results = {
            'events': [],
            'metrics': {},
            'summary': {}
        }

        df = df.copy()
        df['stress_score'] = self.stress_calculator.calculate(df)
        df['state'] = self.state_labeler.label_series(df['stress_score'])

        print(f"Validating against {len(events)} events...")

        for event in events:
            event_result = self._validate_event(
                df,
                event,
                lookback_hours,
                lookahead_hours
            )
            if event_result:
                results['events'].append(event_result)

        if not results['events']:
            print("No valid events found")
            return results

        # Calculate aggregate metrics
        results['metrics'] = self._calculate_metrics(results['events'])
        results['summary'] = self._generate_summary(results['events'], results['metrics'])

        return results

    def _validate_event(
        self,
        df: pd.DataFrame,
        event: Dict,
        lookback_hours: int,
        lookahead_hours: int
    ) -> Optional[Dict]:
        """
        Validate a single event.

        Args:
            df: DataFrame with stress scores
            event: Event dict with 'timestamp'
            lookback_hours: Hours before event
            lookahead_hours: Hours after event

        Returns:
            Event result dict or None if not enough data
        """
        try:
            # Parse event timestamp
            event_time = pd.to_datetime(event['timestamp'])

            # Find candles around event
            event_mask = (df['timestamp'] >= event_time - timedelta(hours=lookback_hours)) & \
                         (df['timestamp'] <= event_time + timedelta(hours=lookahead_hours))
            event_window = df[event_mask]

            if event_window.empty:
                print(f"⚠ No data for event: {event.get('description', 'Unknown')} @ {event_time}")
                return None

            # Find closest candle to event time
            time_diff = (df['timestamp'] - event_time).abs()
            closest_idx = time_diff.idxmin()
            closest_row = df.loc[closest_idx]

            # Calculate pre-event stress (1-4 hours before)
            pre_event_mask = (df['timestamp'] >= event_time - timedelta(hours=lookback_hours)) & \
                             (df['timestamp'] < event_time)
            pre_event_df = df[pre_event_mask]

            if pre_event_df.empty:
                pre_event_stress = None
                pre_event_max_stress = None
            else:
                pre_event_stress = pre_event_df['stress_score'].mean()
                pre_event_max_stress = pre_event_df['stress_score'].max()

            # Calculate event-time stress
            event_stress = closest_row['stress_score']

            # Calculate post-event stress
            post_event_mask = (df['timestamp'] > event_time) & \
                              (df['timestamp'] <= event_time + timedelta(hours=lookahead_hours))
            post_event_df = df[post_event_mask]

            if post_event_df.empty:
                post_event_stress = None
            else:
                post_event_stress = post_event_df['stress_score'].mean()

            result = {
                'event_time': event_time,
                'pair': event.get('pair', 'N/A'),
                'description': event.get('description', 'Unknown'),
                'severity': event.get('severity', 'unknown'),
                'event_stress': event_stress,
                'event_state': closest_row['state'],
                'pre_event_mean_stress': pre_event_stress,
                'pre_event_max_stress': pre_event_max_stress,
                'post_event_mean_stress': post_event_stress,
                'stress_elevated': pre_event_max_stress is not None and pre_event_max_stress > 0.7
            }

            return result

        except Exception as e:
            print(f"Error validating event: {e}")
            return None

    def _calculate_metrics(self, events: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics from validated events.

        Args:
            events: List of event results

        Returns:
            Dictionary with metrics
        """
        events_with_pre = [e for e in events if e['pre_event_max_stress'] is not None]

        if not events_with_pre:
            return {}

        # Stress elevation rate (% events with stress > 0.7 before event)
        elevated_count = sum(1 for e in events_with_pre if e['stress_elevated'])
        stress_elevation_rate = elevated_count / len(events_with_pre)

        # Average stress at event time
        avg_event_stress = np.mean([e['event_stress'] for e in events])

        # Average pre-event stress
        avg_pre_stress = np.mean([e['pre_event_mean_stress'] for e in events_with_pre])

        # Average pre-event max stress
        avg_pre_max_stress = np.mean([e['pre_event_max_stress'] for e in events_with_pre])

        # State distribution at event time
        states_at_event = [e['event_state'] for e in events]
        state_counts = pd.Series(states_at_event).value_counts().to_dict()

        metrics = {
            'total_events_validated': len(events),
            'events_with_pre_data': len(events_with_pre),
            'stress_elevation_rate': stress_elevation_rate,
            'avg_event_stress': avg_event_stress,
            'avg_pre_event_stress': avg_pre_stress,
            'avg_pre_event_max_stress': avg_pre_max_stress,
            'state_distribution_at_event': state_counts
        }

        return metrics

    def _generate_summary(self, events: List[Dict], metrics: Dict) -> Dict:
        """
        Generate human-readable summary.

        Args:
            events: List of event results
            metrics: Calculated metrics

        Returns:
            Summary dict with key findings
        """
        summary = {
            'total_events': len(events),
            'stress_elevation_pct': metrics.get('stress_elevation_rate', 0) * 100,
            'avg_stress_before': metrics.get('avg_pre_event_max_stress', 0),
            'success': None
        }

        # Success criteria from validation plan
        success_criteria = {
            'stress_elevation': metrics.get('stress_elevation_rate', 0) >= 0.7,  # >= 70%
            'low_baseline': metrics.get('avg_pre_event_stress', 0) < 0.3  # < 0.3 average
        }

        summary['success_criteria'] = success_criteria
        summary['success'] = all(success_criteria.values())

        return summary

    def save_results(self, results: Dict, output_file: str):
        """
        Save validation results to JSON file.

        Args:
            results: Results dictionary
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            # Convert datetime objects to strings
            results_serializable = self._serialize_results(results)
            json.dump(results_serializable, f, indent=2)

        print(f"✓ Results saved to {output_file}")

    def _serialize_results(self, results: Dict) -> Dict:
        """Convert datetime objects to strings for JSON serialization."""
        if isinstance(results, dict):
            return {k: self._serialize_results(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._serialize_results(item) for item in results]
        elif isinstance(results, pd.Timestamp):
            return results.isoformat()
        else:
            return results


if __name__ == "__main__":
    # Example usage
    from data.fetch_binance import BinanceFetcher
    from data.tag_events import EventTagger

    # Fetch data
    fetcher = BinanceFetcher()
    df = fetcher.load_data('BTCUSDT', '1h')

    # Load events
    tagger = EventTagger()
    events = tagger.list_events()

    # Run backtest
    backtester = Backtester()
    results = backtester.run_validation(df, events)

    # Print summary
    if results['summary']:
        print("\n=== Validation Summary ===")
        print(f"Events validated: {results['summary']['total_events']}")
        print(f"Stress elevation: {results['summary']['stress_elevation_pct']:.1f}%")
        print(f"Avg stress before events: {results['summary']['avg_stress_before']:.3f}")
        print(f"Success: {results['summary']['success']}")
