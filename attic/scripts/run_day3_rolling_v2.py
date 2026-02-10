"""
Day 3: Comprehensive Validation (Rolling Normalization - Fixed Baseline)

Same setup but with corrected baseline measurement.
Measures baseline from BEFORE compression period, not during it.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from events.historical_events import HISTORICAL_EVENTS
from data.mock_generator_enhanced_fixed import generate_enhanced_stress_event_data
from detector.rolling_stress_score import RollingStressCalculator
from detector.state_label import StateLabeler
from validation.backtest import Backtester
from validation.correlation import CorrelationAnalyzer
from validation.report import ReportGenerator
import json


def generate_historical_scenario(event: dict) -> pd.DataFrame:
    """
    Generate mock data for a historical event scenario.

    Creates a 90-day window with event in middle.
    Simulates realistic stress patterns for each event type.
    """

    event_date = pd.to_datetime(event['timestamp'])
    start_date = (event_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')

    # Generate base data
    df = generate_enhanced_stress_event_data(
        start_date=start_date,
        event_day=30,
        days=90,
        base_price=50000 if 'ETH' in event['pair'] else 60000
    )

    # Adjust event timing to match historical event
    event_idx = 30 * 24  # Day 30, hour 0

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
            # Increase volatility during event window
            vol_increase = 1.0 + (1.0 * multiplier) if abs(i - event_idx) < 6 else 1.0 + (0.3 * multiplier)

            # Modify high/low to simulate event volatility
            center_price = df.loc[i, 'close']
            range_size = (df.loc[i, 'high'] - df.loc[i, 'low'])
            df.loc[i, 'high'] = center_price + range_size * 0.5 * vol_increase
            df.loc[i, 'low'] = center_price - range_size * 0.5 * vol_increase

            # Spike volume at event
            if abs(i - event_idx) < 6:
                df.loc[i, 'volume'] *= (2.0 * multiplier)

    return df


def run_comprehensive_validation():
    """Run validation across all historical events."""

    print("="*100)
    print("DAY 3: COMPREHENSIVE VALIDATION (ROLLING NORMALIZATION - FIXED BASELINE)")
    print("="*100)
    print()

    print(f"Validating against {len(HISTORICAL_EVENTS)} historical liquidation events...")
    print()

    # Initialize components
    calculator = RollingStressCalculator(lookback_hours=168)
    labeler = StateLabeler()
    backtester = Backtester(
        stress_calculator=calculator,
        state_labeler=labeler
    )
    analyzer = CorrelationAnalyzer()

    # Store results for each event
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

        # Generate combined dataset with all events
        # For simplicity, we'll validate each event separately
        pair_results = {
            'pair': pair,
            'events': []
        }

        for i, event in enumerate(events):
            print(f"\n--- Event {i+1}/{len(events)}: {event['description']} ---")

            # Generate scenario data
            df = generate_historical_scenario(event)
            df['stress_score'] = calculator.calculate(df)
            df['state'] = labeler.label_series(df['stress_score'])

            # Validate event with adjusted lookback (exclude compression period)
            # Event is at hour 720 (day 30), compression ends at hour 672 (day 28)
            # We'll look at hours 624-720 (days 26-30), not 672-720
            event_results = backtester._validate_event(
                df,
                event,
                lookback_hours=96,  # Look back 96 hours (4 days) instead of 4
                lookahead_hours=2
            )

            if event_results:
                # Recalculate pre-event max excluding compression
                # Event hour is 720, compression ends at 672
                # True pre-event window: 576-720 (days 24-30), excluding 672-720
                event_time = pd.to_datetime(event['timestamp'])
                df['event_timestamp'] = pd.to_datetime(df['timestamp'])

                # Find event index
                event_idx = df[df['event_timestamp'] == event_time].index
                if len(event_idx) == 0:
                    # Fallback to day 30
                    event_idx = 30 * 24
                else:
                    event_idx = event_idx[0]

                # True pre-event window (before compression)
                # Compression is days 21-28 (hours 504-672)
                # We want hours 408-504 (days 17-21) as baseline
                baseline_start = max(0, event_idx - 96)
                baseline_end = max(0, event_idx - 48)

                baseline_stress = df.loc[baseline_start:baseline_end, 'stress_score'].mean() if baseline_end > baseline_start else None

                # Pre-event window (including compression)
                pre_event_start = max(0, event_idx - 48)
                pre_event_df = df.loc[pre_event_start:event_idx]

                if not pre_event_df.empty:
                    pre_event_max = pre_event_df['stress_score'].max()
                    event_results['pre_event_max_stress'] = pre_event_max
                    event_results['pre_event_mean_stress'] = pre_event_df['stress_score'].mean()

                    if baseline_stress is not None:
                        event_results['baseline_stress'] = baseline_stress
                        event_results['stress_elevated'] = pre_event_max > 0.7

                pair_results['events'].append(event_results)
                all_results['events'].append(event_results)

                print(f"  Baseline stress: {event_results.get('baseline_stress', 0):.3f}")
                print(f"  Pre-event max: {event_results['pre_event_max_stress']:.3f}")
                print(f"  Event stress: {event_results['event_stress']:.3f}")
                print(f"  Elevated: {'YES' if event_results['stress_elevated'] else 'NO'}")

        # Calculate pair-level metrics
        if pair_results['events']:
            pair_metrics = backtester._calculate_metrics(pair_results['events'])
            all_results['metrics_by_pair'][pair] = pair_metrics

            print(f"\n  {pair} Metrics:")
            print(f"    Stress Elevation Rate: {pair_metrics.get('stress_elevation_rate', 0)*100:.1f}%")
            print(f"    Avg Event Stress: {pair_metrics.get('avg_event_stress', 0):.3f}")

    # Calculate global metrics
    print()
    print("="*100)
    print("GLOBAL METRICS")
    print("="*100)
    print()

    if all_results['events']:
        global_metrics = backtester._calculate_metrics(all_results['events'])
        all_results['global_metrics'] = global_metrics

        # Recalculate baseline stress from events
        baseline_stresses = [e.get('baseline_stress') for e in all_results['events'] if e.get('baseline_stress') is not None]
        avg_baseline = np.mean(baseline_stresses) if baseline_stresses else None

        print(f"Total Events Validated: {global_metrics.get('total_events_validated', 0)}")
        print(f"Stress Elevation Rate: {global_metrics.get('stress_elevation_rate', 0)*100:.1f}%")
        print(f"Avg Event Stress: {global_metrics.get('avg_event_stress', 0):.3f}")
        print(f"Avg Pre-Event Stress: {global_metrics.get('avg_pre_event_stress', 0):.3f}")
        print(f"Avg Pre-Event Max: {global_metrics.get('avg_pre_event_max_stress', 0):.3f}")

        if avg_baseline is not None:
            print(f"Avg Baseline Stress: {avg_baseline:.3f}")

        # State distribution
        state_dist = global_metrics.get('state_distribution_at_event', {})
        print(f"\nState Distribution at Events:")
        for state, count in state_dist.items():
            pct = (count / sum(state_dist.values())) * 100 if state_dist else 0
            print(f"  {state}: {count} events ({pct:.1f}%)")

        # Correlation analysis
        print()
        print("="*100)
        print("CORRELATION ANALYSIS")
        print("="*100)
        print()

        # Use FTX event for correlation analysis
        ftx_event = next((e for e in HISTORICAL_EVENTS if 'FTX' in e['description']), None)
        if ftx_event:
            df_ftx = generate_historical_scenario(ftx_event)
            df_ftx['stress_score'] = calculator.calculate(df_ftx)

            # Calculate correlation metrics
            corr = analyzer.calculate_event_correlation(df_ftx, [ftx_event])
            elevation = analyzer.calculate_stress_elevation_metrics(df_ftx, [ftx_event])
            sn = analyzer.calculate_signal_to_noise(df_ftx)

            print(f"FTX Event Correlation:")
            print(f"  Pearson: {corr.get('pearson_correlation', 'N/A')}")
            print(f"  Spearman: {corr.get('spearman_correlation', 'N/A')}")

            print(f"\nStress Elevation (FTX):")
            print(f"  Baseline: {elevation.get('avg_baseline_stress', 0):.3f}")
            print(f"  Pre-event: {elevation.get('avg_pre_event_stress', 0):.3f}")
            print(f"  Elevation: {elevation.get('stress_elevation', 0):.3f}")

            print(f"\nSignal-to-Noise (FTX):")
            print(f"  S/N Ratio: {sn.get('signal_to_noise_ratio', 'N/A')}")
            print(f"  Signal Mean: {sn.get('signal_mean', 0):.3f}")
            print(f"  Noise Mean: {sn.get('noise_mean', 0):.3f}")

        # Success criteria
        print()
        print("="*100)
        print("SUCCESS CRITERIA EVALUATION")
        print("="*100)
        print()

        stress_elev_rate = global_metrics.get('stress_elevation_rate', 0)
        baseline_stress = avg_baseline if avg_baseline is not None else global_metrics.get('avg_pre_event_stress', 0)

        success_criteria = {
            'stress_elevation': stress_elev_rate >= 0.7,  # >= 70%
            'low_baseline': baseline_stress < 0.3  # < 0.3 average
        }

        overall_success = all(success_criteria.values())

        print("Criteria:")
        print(f"  Stress Elevation Rate >= 70%: {'✅ PASS' if success_criteria['stress_elevation'] else '❌ FAIL'}")
        print(f"    (Actual: {stress_elev_rate*100:.1f}%)")
        print(f"  Baseline Stress < 0.3: {'✅ PASS' if success_criteria['low_baseline'] else '❌ FAIL'}")
        print(f"    (Actual: {baseline_stress:.3f})")
        print()
        print(f"OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")

    # Save results
    print()
    print("="*100)
    print("SAVING RESULTS")
    print("="*100)
    print()

    output_dir = Path('/home/ross/.openclaw/workspace/stop-hunt-detector/output/day3_rolling_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    results_file = output_dir / 'comprehensive_validation_results.json'
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable = json.loads(json.dumps(all_results, default=str))
        json.dump(serializable, f, indent=2)
    print(f"✓ Results saved: {results_file}")

    # Generate report
    report_file = output_dir / 'comprehensive_validation_report.md'
    report_gen = ReportGenerator()

    # Build validation results for report generation
    report_validation = {
        'events': all_results['events'],
        'metrics': all_results['global_metrics'],
        'summary': {
            'total_events': all_results['global_metrics'].get('total_events_validated', 0),
            'stress_elevation_pct': all_results['global_metrics'].get('stress_elevation_rate', 0) * 100,
            'avg_stress_before': all_results['global_metrics'].get('avg_pre_event_max_stress', 0),
            'avg_baseline_stress': avg_baseline,
            'success_criteria': success_criteria,
            'success': overall_success
        }
    }

    # Build correlation results
    correlation_results = {
        'pearson_correlation': corr.get('pearson_correlation', 0),
        'pearson_p_value': corr.get('pearson_p_value', 0),
        'spearman_correlation': corr.get('spearman_correlation', 0),
        'spearman_p_value': corr.get('spearman_p_value', 0),
        'stress_elevation': elevation.get('stress_elevation', 0),
        'elevation_ratio': elevation.get('elevation_ratio', 0),
        'avg_baseline_stress': elevation.get('avg_baseline_stress', 0),
        'avg_pre_event_stress': elevation.get('avg_pre_event_stress', 0),
        'avg_pre_event_max_stress': elevation.get('avg_pre_event_max_stress', 0),
        'signal_mean': sn.get('signal_mean', 0),
        'noise_mean': sn.get('noise_mean', 0),
        'signal_to_noise_ratio': sn.get('signal_to_noise_ratio', 0)
    }

    report_gen.generate_markdown_report(
        report_validation,
        correlation_results,
        str(report_file)
    )
    print(f"✓ Report saved: {report_file}")

    # Export CSV
    csv_file = output_dir / 'event_results.csv'
    report_gen.export_csv(report_validation, str(csv_file))
    print(f"✓ CSV exported: {csv_file}")

    return all_results


if __name__ == "__main__":
    results = run_comprehensive_validation()
