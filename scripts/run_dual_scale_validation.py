"""
Dual-Scale Validation (Day 3 Re-Run)

Validates both fast relative stress and slow absolute context.
Uses 2D state table for interpretation.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from events.historical_events import HISTORICAL_EVENTS
from data.mock_generator_enhanced_fixed import generate_enhanced_stress_event_data
from detector.rolling_stress_score import RollingStressCalculator
from detector.slow_context import SlowContextCalculator
from detector.state_label import StateLabeler
import json


def generate_historical_scenario(event: dict) -> pd.DataFrame:
    """
    Generate mock data for a historical event scenario.
    """
    event_date = pd.to_datetime(event['timestamp'])
    start_date = (event_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')

    df = generate_enhanced_stress_event_data(
        start_date=start_date,
        event_day=30,
        days=90,
        base_price=50000 if 'ETH' in event['pair'] else 60000
    )

    # Adjust event timing to match historical event
    event_idx = 30 * 24

    # Add event-specific characteristics based on severity
    severity_multipliers = {
        'high': 1.5,
        'medium': 1.0,
        'low': 0.7
    }
    multiplier = severity_multipliers.get(event['severity'], 1.0)

    # Enhance volatility around event
    for i in range(event_idx - 48, min(event_idx + 24, len(df))):
        if 0 <= i < len(df):
            vol_increase = 1.0 + (1.0 * multiplier) if abs(i - event_idx) < 6 else 1.0 + (0.3 * multiplier)

            center_price = df.loc[i, 'close']
            range_size = (df.loc[i, 'high'] - df.loc[i, 'low'])
            df.loc[i, 'high'] = center_price + range_size * 0.5 * vol_increase
            df.loc[i, 'low'] = center_price - range_size * 0.5 * vol_increase

            if abs(i - event_idx) < 6:
                df.loc[i, 'volume'] *= (2.0 * multiplier)

    return df


def get_2d_state(fast_stress: float, slow_context: float) -> Tuple[str, str, str]:
    """
    Map fast stress and slow context to 2D state table.

    Fast Stress thresholds:
    - Low: < 0.5
    - Rising: 0.5 - 0.7
    - High: > 0.7

    Slow Context thresholds:
    - Cold: < 0.35
    - Neutral: 0.35 - 0.65
    - Hot: > 0.65

    Returns:
        Tuple of (state_fast, state_slow, interpretation)
    """
    # Fast stress categorization
    if fast_stress < 0.5:
        state_fast = 'Low'
    elif fast_stress < 0.7:
        state_fast = 'Rising'
    else:
        state_fast = 'High'

    # Slow context categorization
    if slow_context < 0.35:
        state_slow = 'Cold'
    elif slow_context < 0.65:
        state_slow = 'Neutral'
    else:
        state_slow = 'Hot'

    # 2D state interpretation
    interpretation_map = {
        ('Low', 'Cold'): 'Stable / Ignore',
        ('Low', 'Neutral'): 'Stable / Ignore',
        ('Low', 'Hot'): 'Compression Risk',

        ('Rising', 'Cold'): 'Rising Cold',
        ('Rising', 'Neutral'): 'Watch',
        ('Rising', 'Hot'): 'Elevated Risk',

        ('High', 'Cold'): 'High Cold Clearing',
        ('High', 'Neutral'): 'High',
        ('High', 'Hot'): 'Violent Likely'
    }

    interpretation = interpretation_map.get((state_fast, state_slow), 'Unknown')

    return state_fast, state_slow, interpretation


def run_dual_scale_validation():
    """Run validation with both fast and slow layers."""

    print("="*100)
    print("DUAL-SCALE VALIDATION (FAST + SLOW)")
    print("="*100)
    print()

    print(f"Validating against {len(HISTORICAL_EVENTS)} historical liquidation events...")
    print()

    # Initialize both calculators
    fast_calculator = RollingStressCalculator(lookback_hours=168)
    slow_calculator = SlowContextCalculator(lookback_hours=4380)
    labeler = StateLabeler()

    # Store results
    all_results = {
        'events': [],
        'metrics_by_pair': {},
        'global_metrics': {}
    }

    # Group events by pair
    events_by_pair = {}
    for event in HISTORICAL_EVENTS:
        pair = event['pair']
        if pair not in events_by_pair:
            events_by_pair[pair] = []
        events_by_pair[pair].append(event)

    # Run validation for each pair
    for pair, events in events_by_pair.items():
        print("="*100)
        print(f"VALIDATING PAIR: {pair}")
        print("="*100)
        print(f"Events: {len(events)}")
        print()

        pair_results = {
            'pair': pair,
            'events': []
        }

        for i, event in enumerate(events):
            print(f"\n--- Event {i+1}/{len(events)}: {event['description']} ---")

            # Generate scenario data
            df = generate_historical_scenario(event)

            # Calculate both layers
            df['fast_stress'] = fast_calculator.calculate(df)
            df['slow_context'] = slow_calculator.calculate(df)
            df['state_fast'] = labeler.label_series(df['fast_stress'])

            # Get slow regime labels
            regime_labels = df['slow_context'].apply(lambda x: slow_calculator.get_regime_label(x)[0])
            df['state_slow'] = regime_labels

            # Find event time
            event_time = pd.to_datetime(event['timestamp'])
            df['event_timestamp'] = pd.to_datetime(df['timestamp'])
            event_idx = df[df['event_timestamp'] == event_time].index
            if len(event_idx) == 0:
                event_idx = 30 * 24
            else:
                event_idx = event_idx[0]

            # Event metrics
            event_result = {
                'event_time': event['timestamp'],
                'pair': event['pair'],
                'description': event['description'],
                'severity': event['severity'],
                'fast_stress': df.loc[event_idx, 'fast_stress'],
                'slow_context': df.loc[event_idx, 'slow_context'],
                'state_fast': df.loc[event_idx, 'state_fast'],
                'state_slow': df.loc[event_idx, 'state_slow'],
            }

            # Get 2D state
            state_fast, state_slow, interpretation = get_2d_state(
                event_result['fast_stress'],
                event_result['slow_context']
            )
            event_result['state_fast_2d'] = state_fast
            event_result['state_slow_2d'] = state_slow
            event_result['interpretation'] = interpretation

            # Pre-event fast stress elevation
            pre_event_start = max(0, event_idx - 96)
            pre_event_end = max(0, event_idx - 48)
            pre_event_fast = df.loc[pre_event_start:pre_event_end, 'fast_stress']
            if not pre_event_fast.empty:
                event_result['pre_event_fast_max'] = pre_event_fast.max()
                event_result['pre_event_fast_mean'] = pre_event_fast.mean()
                event_result['fast_elevated'] = event_result['pre_event_fast_max'] > 0.7
            else:
                event_result['pre_event_fast_max'] = np.nan
                event_result['pre_event_fast_mean'] = np.nan
                event_result['fast_elevated'] = False

            # Baseline (before compression)
            baseline_start = max(0, event_idx - 192)
            baseline_end = max(0, event_idx - 144)
            baseline_fast = df.loc[baseline_start:baseline_end, 'fast_stress']
            baseline_slow = df.loc[baseline_start:baseline_end, 'slow_context']

            if not baseline_fast.empty:
                event_result['baseline_fast_mean'] = baseline_fast.mean()
                event_result['baseline_slow_mean'] = baseline_slow.mean()
            else:
                event_result['baseline_fast_mean'] = np.nan
                event_result['baseline_slow_mean'] = np.nan

            pair_results['events'].append(event_result)
            all_results['events'].append(event_result)

            print(f"  Fast stress: {event_result['fast_stress']:.3f}")
            print(f"  Slow context: {event_result['slow_context']:.3f}")
            print(f"  Fast state: {event_result['state_fast']}")
            print(f"  Slow regime: {event_result['state_slow']}")
            print(f"  2D state: {state_fast} × {state_slow} → {interpretation}")
            print(f"  Pre-event fast max: {event_result['pre_event_fast_max']:.3f}")
            print(f"  Baseline fast: {event_result['baseline_fast_mean']:.3f}")
            print(f"  Baseline slow: {event_result['baseline_slow_mean']:.3f}")

        # Calculate pair-level metrics
        if pair_results['events']:
            elevated_count = sum(1 for e in pair_results['events'] if e['fast_elevated'])
            elevation_rate = elevated_count / len(pair_results['events'])

            pair_results['metrics'] = {
                'stress_elevation_rate': elevation_rate,
                'elevated_count': elevated_count,
                'total_events': len(pair_results['events'])
            }

            all_results['metrics_by_pair'][pair] = pair_results['metrics']

            print(f"\n  {pair} Metrics:")
            print(f"    Fast stress elevation rate: {elevation_rate*100:.1f}%")
            print(f"    Elevated events: {elevated_count}/{len(pair_results['events'])}")

    # Calculate global metrics
    print()
    print("="*100)
    print("GLOBAL METRICS")
    print("="*100)
    print()

    if all_results['events']:
        # Fast stress elevation
        elevated_count = sum(1 for e in all_results['events'] if e['fast_elevated'])
        elevation_rate = elevated_count / len(all_results['events'])

        # Slow context distribution at events
        slow_contexts = [e['slow_context'] for e in all_results['events']]
        avg_slow_context = np.mean(slow_contexts)

        # Fast stress at events
        fast_stresses = [e['fast_stress'] for e in all_results['events']]
        avg_fast_stress = np.mean(fast_stresses)

        all_results['global_metrics'] = {
            'total_events_validated': len(all_results['events']),
            'stress_elevation_rate': elevation_rate,
            'elevated_count': elevated_count,
            'avg_fast_stress': avg_fast_stress,
            'avg_slow_context': avg_slow_context,
            'slow_context_distribution': {
                'cold': sum(1 for e in all_results['events'] if e['state_slow'] == 'COLD'),
                'neutral': sum(1 for e in all_results['events'] if e['state_slow'] == 'NEUTRAL'),
                'hot': sum(1 for e in all_results['events'] if e['state_slow'] == 'HOT')
            },
            'fast_stress_distribution': {
                'low': sum(1 for e in all_results['events'] if e['state_fast'] == 'NORMAL'),
                'stressed': sum(1 for e in all_results['events'] if e['state_fast'] == 'STRESSED'),
                'fragile': sum(1 for e in all_results['events'] if e['state_fast'] == 'FRAGILE'),
                'imminent': sum(1 for e in all_results['events'] if e['state_fast'] == 'IMMINENT_CLEARING')
            },
            'state_2d_distribution': {}
        }

        # 2D state distribution
        for event in all_results['events']:
            state_2d = (event['state_fast_2d'], event['state_slow_2d'])
            state_2d_str = f"{state_2d[0]} × {state_2d[1]}"
            all_results['global_metrics']['state_2d_distribution'][state_2d_str] = \
                all_results['global_metrics']['state_2d_distribution'].get(state_2d_str, 0) + 1

        print(f"Total Events Validated: {len(all_results['events'])}")
        print(f"Fast Stress Elevation Rate: {elevation_rate*100:.1f}%")
        print(f"Avg Fast Stress at Events: {avg_fast_stress:.3f}")
        print(f"Avg Slow Context at Events: {avg_slow_context:.3f}")
        print()
        print("Slow Context Distribution at Events:")
        for regime, count in all_results['global_metrics']['slow_context_distribution'].items():
            pct = (count / len(all_results['events'])) * 100
            print(f"  {regime}: {count} events ({pct:.1f}%)")
        print()
        print("Fast Stress Distribution at Events:")
        for state, count in all_results['global_metrics']['fast_stress_distribution'].items():
            pct = (count / len(all_results['events'])) * 100
            print(f"  {state}: {count} events ({pct:.1f}%)")
        print()
        print("2D State Distribution at Events:")
        for state_2d, count in all_results['global_metrics']['state_2d_distribution'].items():
            pct = (count / len(all_results['events'])) * 100
            print(f"  {state_2d[0]} × {state_2d[1]}: {count} events ({pct:.1f}%)")

    # Save results
    print()
    print("="*100)
    print("SAVING RESULTS")
    print("="*100)
    print()

    output_dir = Path('/home/ross/.openclaw/workspace/stop-hunt-detector/output/dual_scale')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    results_file = output_dir / 'dual_scale_results.json'
    with open(results_file, 'w') as f:
        serializable = json.loads(json.dumps(all_results, default=str))
        json.dump(serializable, f, indent=2)
    print(f"✓ Results saved: {results_file}")

    # Save CSV
    csv_file = output_dir / 'event_details.csv'
    csv_data = []
    for event in all_results['events']:
        csv_data.append({
            'timestamp': event['event_time'],
            'pair': event['pair'],
            'description': event['description'],
            'severity': event['severity'],
            'fast_stress': event['fast_stress'],
            'slow_context': event['slow_context'],
            'state_fast': event['state_fast'],
            'state_slow': event['state_slow'],
            'state_fast_2d': event['state_fast_2d'],
            'state_slow_2d': event['state_slow_2d'],
            'interpretation': event['interpretation'],
            'pre_event_fast_max': event['pre_event_fast_max'],
            'baseline_fast_mean': event['baseline_fast_mean'],
            'baseline_slow_mean': event['baseline_slow_mean'],
            'fast_elevated': event['fast_elevated']
        })

    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(csv_file, index=False)
    print(f"✓ CSV exported: {csv_file}")

    return all_results


if __name__ == "__main__":
    results = run_dual_scale_validation()
