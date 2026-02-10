"""
Stress Score Visualization

Generate plots of stress scores, individual signals, and events.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def plot_stress_score(
    df: pd.DataFrame,
    output_file: str,
    title: str = "Market Stress Score",
    show_events: bool = True
):
    """
    Plot stress score over time with state coloring.

    Args:
        df: DataFrame with timestamp, stress_score, state columns
        output_file: Output file path
        title: Plot title
        show_events: Whether to mark high-stress events
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Define state colors
    state_colors = {
        'NORMAL': 'green',
        'STRESSED': 'yellow',
        'FRAGILE': 'orange',
        'IMMINENT_CLEARING': 'red'
    }

    # Apply state colors
    df['color'] = df['state'].map(state_colors)

    # Plot 1: Stress score with state background
    ax1.plot(df['timestamp'], df['stress_score'], linewidth=1.5, color='blue', alpha=0.7, label='Stress Score')

    # Add state regions
    for state, color in state_colors.items():
        state_data = df[df['state'] == state]
        if not state_data.empty:
            ax1.scatter(
                state_data['timestamp'],
                state_data['stress_score'],
                c=color,
                alpha=0.3,
                s=10,
                label=state
            )

    # Add threshold lines
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High-Stress Threshold (0.7)')
    ax1.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='Stressed Threshold (0.3)')

    ax1.set_ylabel('Stress Score (0.0 - 1.0)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Individual signals
    signal_cols = ['signal_volatility', 'signal_liquidity', 'signal_continuation', 'signal_speed']
    signal_names = ['Volatility', 'Liquidity', 'Continuation', 'Speed']
    colors = ['purple', 'cyan', 'magenta', 'lime']

    for col, name, color in zip(signal_cols, signal_names, colors):
        if col in df.columns:
            ax2.plot(df['timestamp'], df[col], linewidth=1, alpha=0.7, color=color, label=name)

    ax2.set_ylabel('Signal Value (0.0 - 1.0)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Mark high-stress periods
    if show_events:
        high_stress_mask = df['stress_score'] >= 0.7
        high_stress_timestamps = df[high_stress_mask]['timestamp']
        ax1.scatter(high_stress_timestamps, df.loc[high_stress_mask, 'stress_score'],
                   c='red', s=50, marker='v', alpha=0.7, zorder=5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Stress score plot saved: {output_file}")


def plot_signal_distribution(df: pd.DataFrame, output_file: str):
    """
    Plot distribution of individual signals.

    Args:
        df: DataFrame with signal columns
        output_file: Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    signal_cols = ['signal_volatility', 'signal_liquidity', 'signal_continuation', 'signal_speed']
    signal_names = ['Volatility Compression', 'Liquidity Fragility', 'Continuation Failure', 'Speed Asymmetry']

    for idx, (col, name) in enumerate(zip(signal_cols, signal_names)):
        ax = axes[idx]

        if col in df.columns:
            data = df[col].dropna()

            # Histogram
            ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()

            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

            ax.set_xlabel('Signal Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Signal distribution plot saved: {output_file}")


def plot_stress_heatmap(df: pd.DataFrame, output_file: str):
    """
    Plot stress score heatmap (stress by hour/day).

    Args:
        df: DataFrame with timestamp and stress_score columns
        output_file: Output file path
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Create pivot table
    pivot = df.pivot_table(
        values='stress_score',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )

    # Day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot.columns = day_labels

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(day_labels)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(day_labels)
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Average Stress Score', rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(day_labels)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Stress Score Heatmap by Hour and Day', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Hour of Day', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Stress heatmap saved: {output_file}")


def plot_event_validation(
    df: pd.DataFrame,
    events: pd.DataFrame,
    output_file: str,
    lookback_hours: int = 4
):
    """
    Plot stress scores around validation events.

    Args:
        df: DataFrame with timestamp and stress_score
        events: DataFrame with event timestamps
        output_file: Output file path
        lookback_hours: Hours before event to show
    """
    fig, axes = plt.subplots(len(events), 1, figsize=(16, 4 * len(events)))

    if len(events) == 0:
        print("No events to plot")
        return

    if len(events) == 1:
        axes = [axes]

    for idx, (_, event) in enumerate(events.iterrows()):
        ax = axes[idx]

        event_time = pd.to_datetime(event['timestamp'])

        # Get window around event
        start_time = event_time - pd.Timedelta(hours=lookback_hours)
        end_time = event_time + pd.Timedelta(hours=2)

        window_mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        window_data = df[window_mask].copy()

        # Plot stress score
        ax.plot(window_data['timestamp'], window_data['stress_score'],
                linewidth=2, color='blue', label='Stress Score')

        # Mark event time
        ax.axvline(x=event_time, color='red', linestyle='--', linewidth=2, label='Event Time')
        ax.axhline(y=0.7, color='red', linestyle=':', alpha=0.5)

        # Highlight pre-event period
        pre_event_start = event_time - pd.Timedelta(hours=4)
        ax.axvspan(pre_event_start, event_time, alpha=0.2, color='yellow', label='Pre-Event (4h)')

        ax.set_ylabel('Stress Score', fontsize=10)
        ax.set_title(f"Event: {event.get('description', 'Unknown')} @ {event_time.strftime('%Y-%m-%d %H:%M')}",
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Event validation plot saved: {output_file}")


def generate_all_visualizations(
    df: pd.DataFrame,
    output_dir: str,
    title: str = "Market Stress Score Analysis"
):
    """
    Generate all visualizations.

    Args:
        df: DataFrame with stress scores and signals
        output_dir: Output directory
        title: Analysis title
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Main stress score plot
    plot_stress_score(
        df,
        output_path / f'stress_score_{timestamp}.png',
        title=title
    )

    # Signal distribution
    plot_signal_distribution(
        df,
        output_path / f'signal_distribution_{timestamp}.png'
    )

    # Stress heatmap
    plot_stress_heatmap(
        df,
        output_path / f'stress_heatmap_{timestamp}.png'
    )

    print(f"\n✓ All visualizations generated in: {output_path}")


if __name__ == "__main__":
    # Example usage
    from data.mock_generator import generate_stress_event_data
    from detector.stress_score import StressCalculator
    from detector.state_label import StateLabeler

    # Generate test data
    df = generate_stress_event_data(event_day=30, days=60)

    # Calculate stress
    calculator = StressCalculator()
    df['stress_score'] = calculator.calculate(df)

    # Add signals
    df['signal_volatility'] = calculator.volatility.calculate(df)
    df['signal_liquidity'] = calculator.liquidity.calculate(df)
    df['signal_continuation'] = calculator.continuation.calculate(df)
    df['signal_speed'] = calculator.speed.calculate(df)

    # Label states
    labeler = StateLabeler()
    df['state'] = labeler.label_series(df['stress_score'])

    # Generate visualizations
    generate_all_visualizations(
        df,
        '/home/ross/.openclaw/workspace/stop-hunt-detector/output/visualizations',
        title="Mock Data Stress Analysis"
    )
