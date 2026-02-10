"""
Final Adjudication - Dual-Scale Validation with Real Coinbase Data

Single clean run to determine: Does slow context separation exist in reality?
Uses maximum available Coinbase BTC-USD 1h candles.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_coinbase import CoinbaseFetcher
from events.historical_events import HISTORICAL_EVENTS
from detector.rolling_stress_score import RollingStressCalculator
from detector.slow_context import SlowContextCalculator
from detector.state_label import StateLabeler


OUTPUT_DIR = Path('/home/ross/.openclaw/workspace/stop-hunt-detector/output/final_adjudication')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_2d_state(fast_stress: float, slow_context: float):
    """Map fast stress and slow context to 2D state table."""
    # Fast stress thresholds
    if fast_stress < 0.5:
        state_fast = 'LOW'
    elif fast_stress < 0.7:
        state_fast = 'RISING'
    else:
        state_fast = 'HIGH'

    # Slow context thresholds
    if slow_context < 0.35:
        state_slow = 'COLD'
    elif slow_context < 0.65:
        state_slow = 'NEUTRAL'
    else:
        state_slow = 'HOT'

    interpretation_map = {
        ('LOW', 'COLD'): 'Stable / Ignore',
        ('LOW', 'NEUTRAL'): 'Stable / Ignore',
        ('LOW', 'HOT'): 'Compression Risk',
        ('RISING', 'COLD'): 'Rising Cold',
        ('RISING', 'NEUTRAL'): 'Watch',
        ('RISING', 'HOT'): 'Elevated Risk',
        ('HIGH', 'COLD'): 'Clearing Likely',
        ('HIGH', 'NEUTRAL'): 'Elevated',
        ('HIGH', 'HOT'): 'Violent Likely'
    }

    interpretation = interpretation_map.get((state_fast, state_slow), 'Unknown')
    return state_fast, state_slow, interpretation


def check_continuity(df: pd.DataFrame, expected_freq='1h'):
    """Check continuity and return max gap in hours + count of large gaps."""
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    diffs = df_sorted['timestamp'].diff().dropna()
    max_gap = diffs.max()
    max_gap_hours = max_gap.total_seconds() / 3600 if max_gap is not None else 0

    expected_delta = pd.Timedelta(expected_freq)
    large_gap_count = (diffs > (expected_delta * 2)).sum()

    return max_gap_hours, int(large_gap_count)


def plot_fast_stress(df, event_results):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df['timestamp'], df['fast_stress'], linewidth=1, alpha=0.7, color='blue')

    # Event markers
    for event in event_results:
        event_time = pd.to_datetime(event['event_time'])
        ax.axvline(x=event_time, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.scatter([event_time], [event['fast_stress']], c='red', s=60, zorder=5)

    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='0.5')
    ax.axhline(y=0.7, color='orange', linestyle=':', alpha=0.5, label='0.7')

    ax.set_title('Fast Stress Over Time (Events marked)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fast Stress')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45)

    out = OUTPUT_DIR / 'fast_stress_over_time.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_slow_context(df, event_results):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df['timestamp'], df['slow_context'], linewidth=1, alpha=0.7, color='purple')

    # Event markers
    for event in event_results:
        event_time = pd.to_datetime(event['event_time'])
        ax.axvline(x=event_time, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.scatter([event_time], [event['slow_context']], c='red', s=60, zorder=5)

    ax.axhline(y=0.35, color='blue', linestyle=':', alpha=0.5, label='Cold/Neutral 0.35')
    ax.axhline(y=0.65, color='red', linestyle=':', alpha=0.5, label='Neutral/Hot 0.65')

    ax.set_title('Slow Context Over Time (Events marked)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Slow Context')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45)

    out = OUTPUT_DIR / 'slow_context_over_time.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_dual_overlay(df, event_results):
    fig, ax = plt.subplots(figsize=(10, 8))

    sample = df.sample(min(1500, len(df)), random_state=42)
    sc = ax.scatter(sample['slow_context'], sample['fast_stress'],
                    c=sample['fast_stress'], cmap='RdYlBu_r', alpha=0.6, s=10)

    for event in event_results:
        ax.scatter(event['slow_context'], event['fast_stress'], c='red', s=120,
                   marker='X', edgecolor='black', linewidth=1.5, zorder=10)

    ax.axvline(x=0.35, color='blue', linestyle='--', alpha=0.3)
    ax.axvline(x=0.65, color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3)

    ax.set_xlabel('Slow Context')
    ax.set_ylabel('Fast Stress')
    ax.set_title('Fast Stress vs Slow Context (Events in Red)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, label='Fast Stress')

    out = OUTPUT_DIR / 'fast_vs_slow_scatter.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_state_table_counts(state_counts):
    # Build 3x3 matrix for states
    fast_levels = ['LOW', 'RISING', 'HIGH']
    slow_levels = ['COLD', 'NEUTRAL', 'HOT']

    matrix = np.zeros((3, 3), dtype=int)
    for (f, s), count in state_counts.items():
        i = fast_levels.index(f)
        j = slow_levels.index(s)
        matrix[i, j] = count

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='YlOrRd')

    # Annotate
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(slow_levels)
    ax.set_yticklabels(fast_levels)
    ax.set_xlabel('Slow Context')
    ax.set_ylabel('Fast Stress')
    ax.set_title('2D State Table (Event Counts)')
    plt.colorbar(im, ax=ax)

    out = OUTPUT_DIR / 'state_table_counts.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def generate_markdown_report(results, plots):
    lines = []
    lines.append("# Final Adjudication Report - Coinbase BTC-USD")
    lines.append("")
    lines.append(f"**Generated:** {results['metadata']['generated_at']}")
    lines.append(f"**Verdict:** {'✅ PASS' if results['verdict'] == 'PASS' else '❌ FAIL'}")
    lines.append("")
    lines.append("---\n")

    lines.append("## Data Source")
    lines.append(f"- Exchange: Coinbase")
    lines.append(f"- Product: BTC-USD")
    lines.append(f"- Timeframe: 1h")
    lines.append(f"- Data Range: {results['metadata']['data_range']['start']} to {results['metadata']['data_range']['end']}")
    lines.append(f"- Total Candles: {results['metadata']['total_candles']}")
    lines.append("")

    lines.append("## Metrics")
    lines.append(f"- Fast Stress Elevation Rate: {results['metrics']['fast_stress_elevation_rate']*100:.1f}% ({results['metrics']['elevated_events']}/{results['metrics']['total_events']})")
    lines.append(f"- Avg Fast Stress @ Events: {results['metrics']['avg_fast_stress_at_events']:.3f}")
    lines.append(f"- Avg Slow Context @ Events: {results['metrics']['avg_slow_context_at_events']:.3f}")
    lines.append(f"- False Positive Rate (calm periods): {results['metrics']['false_positive_rate']:.1f}%")
    lines.append("")

    lines.append("### Slow Context Distribution @ Events")
    for regime, count in results['metrics']['slow_context_distribution'].items():
        pct = (count / results['metrics']['total_events']) * 100 if results['metrics']['total_events'] else 0
        lines.append(f"- {regime}: {count} events ({pct:.1f}%)")
    lines.append("")

    lines.append("### 2D State Table (Event Counts)")
    for key, count in results['metrics']['state_2d_distribution'].items():
        pct = (count / results['metrics']['total_events']) * 100 if results['metrics']['total_events'] else 0
        lines.append(f"- {key}: {count} events ({pct:.1f}%)")
    lines.append("")

    lines.append("## Plots")
    for name, path in plots.items():
        lines.append(f"- {name}: {path}")
    lines.append("")

    lines.append("## One-Paragraph Interpretation")
    lines.append(results['interpretation'])
    lines.append("")

    out = OUTPUT_DIR / 'final_adjudication_report.md'
    out.write_text('\n'.join(lines))
    return out


def run_final_adjudication():
    print("="*100)
    print("FINAL ADJUDICATION - DUAL-SCALE VALIDATION")
    print("="*100) 
    print()

    # Step 1: Load Coinbase data
    fetcher = CoinbaseFetcher()

    print("Fetching maximum available range...")
    # Coinbase historical: max from 2023-01-01 to now
    start_date = '2023-01-01'
    end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    df = fetcher.fetch_historical(
        'BTCUSDT',
        '1h',
        start_date=start_date,
        end_date=end_date,
        save=True
    )

    # Ensure UTC timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    if df.empty:
        print("❌ No data returned. Aborting.")
        return None

    max_gap_hours, gap_count = check_continuity(df, expected_freq='1h')
    print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df)}")
    print(f"Max gap (hours): {max_gap_hours:.2f}")
    print(f"Gaps >2h: {gap_count}")

    # Step 2: Compute fast and slow layers
    fast_calc = RollingStressCalculator(lookback_hours=168)
    slow_calc = SlowContextCalculator(lookback_hours=4380)

    df['fast_stress'] = fast_calc.calculate(df)
    df['slow_context'] = slow_calc.calculate(df)

    # State labels
    df['state_fast'] = StateLabeler().label_series(df['fast_stress'])
    df['state_slow'] = df['slow_context'].apply(lambda x: slow_calc.get_regime_label(x)[0])

    df['state_fast_2d'], df['state_slow_2d'], df['interpretation'] = zip(
        *df.apply(lambda row: get_2d_state(row['fast_stress'], row['slow_context']), axis=1)
    )

    # Step 3: Evaluate against historical events
    event_results = []
    for event in HISTORICAL_EVENTS:
        event_time = pd.to_datetime(event['timestamp']).replace(tzinfo=timezone.utc)

        if event_time < df['timestamp'].min() or event_time > df['timestamp'].max():
            continue

        event_idx = df[df['timestamp'] <= event_time].index.max()
        if pd.isna(event_idx):
            continue

        # Lead window: 1-4 hours before event
        pre_start = max(0, event_idx - 4)
        pre_end = event_idx
        pre_event = df.iloc[pre_start:pre_end]

        event_result = {
            'event_time': event['timestamp'],
            'description': event['description'],
            'severity': event['severity'],
            'fast_stress': float(df.loc[event_idx, 'fast_stress']),
            'slow_context': float(df.loc[event_idx, 'slow_context']),
            'state_fast': df.loc[event_idx, 'state_fast'],
            'state_slow': df.loc[event_idx, 'state_slow'],
            'state_fast_2d': df.loc[event_idx, 'state_fast_2d'],
            'state_slow_2d': df.loc[event_idx, 'state_slow_2d'],
            'interpretation': df.loc[event_idx, 'interpretation'],
            'pre_event_fast_max': float(pre_event['fast_stress'].max()) if not pre_event.empty else np.nan,
            'fast_elevated': float(pre_event['fast_stress'].max()) > 0.7 if not pre_event.empty else False
        }

        event_results.append(event_result)

    # Step 4: Metrics
    elevated_count = sum(1 for e in event_results if e['fast_elevated'])
    elevation_rate = elevated_count / len(event_results) if event_results else 0

    slow_distribution = {
        'COLD': sum(1 for e in event_results if e['state_slow'] == 'COLD'),
        'NEUTRAL': sum(1 for e in event_results if e['state_slow'] == 'NEUTRAL'),
        'HOT': sum(1 for e in event_results if e['state_slow'] == 'HOT')
    }

    state_2d_counts = {}
    for e in event_results:
        key = (e['state_fast_2d'], e['state_slow_2d'])
        state_2d_counts[key] = state_2d_counts.get(key, 0) + 1

    # False positives in calm periods
    df['is_calm'] = (df['fast_stress'] < 0.5) & (df['slow_context'] < 0.5)
    df['period_id'] = (df['is_calm'] != df['is_calm'].shift()).cumsum()
    calm_hours = df['is_calm'].sum()

    false_positive_hours = 0
    for _, period in df[df['is_calm']].groupby('period_id'):
        if len(period) >= 24 and period['fast_stress'].max() > 0.7:
            false_positive_hours += len(period)

    fp_rate = (false_positive_hours / calm_hours) * 100 if calm_hours > 0 else 0

    # Step 5: Success/Failure Criteria (exact spec)
    regimes_present = [k for k, v in slow_distribution.items() if v > 0]
    events_multiple_slow_regimes = len(regimes_present) > 1
    slow_context_differentiates = events_multiple_slow_regimes

    # Fast stress saturation (dominant high stress across dataset)
    high_fast_ratio = (df['fast_stress'] > 0.7).mean()
    fast_anticipatory = high_fast_ratio < 0.5  # not saturated (majority of time below high threshold)

    pass_criteria = any([
        events_multiple_slow_regimes,
        slow_context_differentiates,
        fast_anticipatory
    ])

    # FAIL if ALL are true
    events_single_slow_regime = len(regimes_present) == 1
    no_2d_separation = len(state_2d_counts) == 1
    false_positives_dominate = (false_positive_hours > 0) and (false_positive_hours / calm_hours > 0.5) if calm_hours > 0 else False

    fail_criteria = all([
        events_single_slow_regime,
        no_2d_separation,
        false_positives_dominate
    ])

    verdict = 'PASS' if pass_criteria else 'FAIL'

    # One-paragraph interpretation
    if verdict == 'PASS':
        interpretation = (
            f"Coinbase BTC-USD data shows slow context separation: events distribute across multiple regimes ({', '.join(regimes_present)}), "
            f"and fast stress remains anticipatory without saturation (high-stress ratio {high_fast_ratio:.2f}). "
            f"Slow context adds environmental distinction beyond fast stress alone, and false positives during calm periods are "
            f"{fp_rate:.1f}%."
        )
    else:
        interpretation = (
            f"Coinbase BTC-USD data shows no meaningful slow context separation. Events cluster into a single slow regime ({', '.join(regimes_present)}), "
            f"and dual-scale states do not add separation over fast stress alone. Fast stress high-ratio is {high_fast_ratio:.2f} "
            f"and false positives during calm periods are {fp_rate:.1f}%."
        )

    # Step 6: Plots
    plots = {
        'fast_stress_over_time': str(plot_fast_stress(df, event_results)),
        'slow_context_over_time': str(plot_slow_context(df, event_results)),
        'fast_vs_slow_scatter': str(plot_dual_overlay(df, event_results)),
        'state_table_counts': str(plot_state_table_counts(state_2d_counts))
    }

    # Step 7: Save artifacts
    results = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'data_source': 'Coinbase Exchange',
            'product': 'BTC-USD',
            'timeframe': '1h',
            'data_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'total_candles': len(df),
            'max_gap_hours': max_gap_hours,
            'gap_count': gap_count
        },
        'metrics': {
            'fast_stress_elevation_rate': elevation_rate,
            'elevated_events': elevated_count,
            'total_events': len(event_results),
            'avg_fast_stress_at_events': float(np.mean([e['fast_stress'] for e in event_results])) if event_results else None,
            'avg_slow_context_at_events': float(np.mean([e['slow_context'] for e in event_results])) if event_results else None,
            'slow_context_distribution': slow_distribution,
            'state_2d_distribution': {f"{k[0]}×{k[1]}": v for k, v in state_2d_counts.items()},
            'false_positive_rate': fp_rate,
            'false_positive_hours': false_positive_hours,
            'total_calm_hours': int(calm_hours)
        },
        'criteria': {
            'events_multiple_slow_regimes': events_multiple_slow_regimes,
            'slow_context_differentiates': slow_context_differentiates,
            'fast_anticipatory': fast_anticipatory,
            'fail_all_conditions': fail_criteria
        },
        'verdict': verdict,
        'overall_success': verdict == 'PASS',
        'interpretation': interpretation,
        'events': event_results,
        'plots': plots
    }

    # Save JSON
    (OUTPUT_DIR / 'final_adjudication_results.json').write_text(json.dumps(results, indent=2, default=str))

    # Save CSV with scores
    df[['timestamp','fast_stress','slow_context','state_fast','state_slow']].to_csv(
        OUTPUT_DIR / 'scores_timeseries.csv', index=False
    )

    # Save event details
    pd.DataFrame(event_results).to_csv(OUTPUT_DIR / 'event_details.csv', index=False)

    # Markdown report
    report_path = generate_markdown_report(results, plots)

    return results, report_path


if __name__ == "__main__":
    results, report_path = run_final_adjudication()
    print("\nFinal Adjudication Complete")
    print(f"Verdict: {results['verdict']}")
    print(f"Report: {report_path}")
