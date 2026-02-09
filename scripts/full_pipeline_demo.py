"""
Full Pipeline Demo

Demonstrates the complete stress detector pipeline:
1. Data generation/loading
2. Signal calculation
3. Stress scoring
4. State labeling
5. Validation against tagged events
6. Report generation
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mock_generator import generate_stress_event_data
from detector.stress_score import StressCalculator
from detector.state_label import StateLabeler
from validation.backtest import Backtester
from validation.correlation import CorrelationAnalyzer
from validation.report import ReportGenerator


def run_full_pipeline():
    """Run complete stress detector pipeline."""

    print("="*80)
    print("STOP-HUNT DETECTOR - FULL PIPELINE DEMO")
    print("="*80)
    print()

    # Step 1: Generate data with simulated stress event
    print("STEP 1: Generating synthetic data with stress event...")
    df = generate_stress_event_data(event_day=30, days=60)
    print(f"  ✓ Generated {len(df)} candles")
    print(f"  ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    # Step 2: Calculate stress scores
    print("STEP 2: Calculating stress scores...")
    calculator = StressCalculator()
    stress_score, details = calculator.calculate_with_details(df)

    # Add all columns to DataFrame
    df['stress_score'] = stress_score
    for col in details.columns:
        df[col] = details[col]

    print(f"  ✓ Stress scores calculated")
    print(f"  ✓ Latest score: {stress_score.iloc[-1]:.3f}")
    print()

    # Step 3: Label states
    print("STEP 3: Labeling market states...")
    labeler = StateLabeler()
    df['state'] = labeler.label_series(stress_score)
    print(f"  ✓ States labeled")

    # Get state summary
    summary = labeler.get_state_summary(df)
    print(f"\n  State Distribution:")
    for state, info in summary.items():
        print(f"    {info['emoji']} {state}: {info['count']} candles ({info['percentage']}%)")
    print()

    # Step 4: Create tagged events for validation
    print("STEP 4: Creating validation events...")
    events = [
        {
            'timestamp': df['timestamp'].iloc[30*24].isoformat(),  # Event day 30
            'pair': 'BTCUSDT',
            'description': 'Simulated stop hunt event',
            'severity': 'high'
        }
    ]
    print(f"  ✓ Created {len(events)} validation event(s)")
    print()

    # Step 5: Run backtest
    print("STEP 5: Running historical validation...")
    backtester = Backtester(
        stress_calculator=calculator,
        state_labeler=labeler
    )

    validation_results = backtester.run_validation(
        df,
        events,
        lookback_hours=4,
        lookahead_hours=2
    )
    print(f"  ✓ Validation complete")

    if validation_results['events']:
        event_result = validation_results['events'][0]
        print(f"\n  Event Results:")
        print(f"    Event stress: {event_result['event_stress']:.3f}")
        print(f"    Pre-event max stress: {event_result['pre_event_max_stress']:.3f}")
        print(f"    Stress elevated: {'YES' if event_result['stress_elevated'] else 'NO'}")
    print()

    # Step 6: Correlation analysis
    print("STEP 6: Running correlation analysis...")
    analyzer = CorrelationAnalyzer()

    correlation_results = analyzer.calculate_event_correlation(
        df,
        events,
        lookback_hours=4
    )
    print(f"  ✓ Correlation complete")
    print(f"    Pearson: {correlation_results['pearson_correlation']:.3f}")
    print(f"    Spearman: {correlation_results['spearman_correlation']:.3f}")
    print()

    # Stress elevation
    elevation = analyzer.calculate_stress_elevation_metrics(df, events)
    print(f"  Stress Elevation:")
    print(f"    Baseline: {elevation['avg_baseline_stress']:.3f}")
    print(f"    Pre-event: {elevation['avg_pre_event_stress']:.3f}")
    print(f"    Elevation: {elevation['stress_elevation']:.3f}")
    print()

    # Signal-to-noise
    sn = analyzer.calculate_signal_to_noise(df)
    print(f"  Signal-to-Noise:")
    print(f"    Signal mean: {sn['signal_mean']:.3f}")
    print(f"    Noise mean: {sn['noise_mean']:.3f}")
    print(f"    S/N ratio: {sn['signal_to_noise_ratio']:.3f}")
    print()

    # Step 7: Generate report
    print("STEP 7: Generating validation report...")
    report_gen = ReportGenerator()

    # Create output directory
    output_dir = Path('/home/ross/.openclaw/workspace/stop-hunt-detector/output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / 'validation_report.md'
    report_gen.generate_markdown_report(
        validation_results,
        {**correlation_results, **elevation, **sn},
        str(report_file)
    )
    print(f"  ✓ Report generated: {report_file}")
    print()

    # Step 8: Summary
    print("="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print()

    if validation_results['summary']:
        summary = validation_results['summary']
        print(f"Events Validated: {summary['total_events']}")
        print(f"Stress Elevation Rate: {summary.get('stress_elevation_pct', 0):.1f}%")
        print(f"Avg Stress Before Events: {summary.get('avg_stress_before', 0):.3f}")
        print()

        success_criteria = summary.get('success_criteria', {})
        print("Success Criteria:")
        for criteria_name, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criteria_name.replace('_', ' ').title()}: {status}")
        print()

        result_status = "✅ SUCCESS" if summary.get('success') else "❌ FAILURE"
        print(f"Overall Result: {result_status}")
    else:
        print("No validation summary available")

    print()
    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    return validation_results, correlation_results


if __name__ == "__main__":
    validation_results, correlation_results = run_full_pipeline()
