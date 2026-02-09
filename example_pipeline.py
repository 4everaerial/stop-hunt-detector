#!/usr/bin/env python3
"""
Example pipeline demonstrating stop-hunt detector usage.

This script demonstrates:
1. Data fetching from Binance
2. Volatility signal calculation
3. Stress score composition
4. State labeling
5. Output visualization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from signals.volatility import VolatilitySignal
from signals.liquidity import LiquiditySignal
from signals.continuation import ContinuationSignal
from signals.speed import SpeedSignal
from detector.stress_score import StressScoreDetector
from detector.state_label import StateLabeler


def main():
    """Run the example pipeline."""

    print("\n" + "="*60)
    print("Stop-Hunt Detector - Example Pipeline")
    print("="*60)

    # For this example, we'll use mock data instead of fetching from Binance
    # In production, use: df = fetch_historical_data('BTCUSDT', '1h', days=30)
    print("\nStep 1: Load OHLCV data")
    print("Note: Using mock data for demonstration. Use fetch_binance.py for real data.")

    # Generate mock data
    import numpy as np
    np.random.seed(42)
    n = 100

    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1h'),
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.1) + np.random.rand(n) * 2,
        'low': 100 + np.cumsum(np.random.randn(n) * 0.1) - np.random.rand(n) * 2,
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': 1000 + np.random.rand(n) * 100
    }

    df = pd.DataFrame(data)
    # Ensure high >= close and low <= close
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)

    print(f"✓ Loaded {len(df)} candles")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Calculate individual stress signals
    print("\nStep 2: Calculate stress signals")

    print("  - Volatility compression signal...")
    vol_signal = VolatilitySignal()
    df = vol_signal.calculate(df)
    print(f"✓ Volatility score range: [{df['volatility_score'].min():.3f}, {df['volatility_score'].max():.3f}]")

    print("  - Liquidity fragility signal...")
    liq_signal = LiquiditySignal()
    df = liq_signal.calculate(df)
    print(f"✓ Liquidity score range: [{df['liquidity_score'].min():.3f}, {df['liquidity_score'].max():.3f}]")

    print("  - Continuation failure signal...")
    cont_signal = ContinuationSignal()
    df = cont_signal.calculate(df)
    print(f"✓ Continuation score range: [{df['continuation_score'].min():.3f}, {df['continuation_score'].max():.3f}]")

    print("  - Speed asymmetry signal...")
    speed_signal = SpeedSignal()
    df = speed_signal.calculate(df)
    print(f"✓ Speed score range: [{df['speed_score'].min():.3f}, {df['speed_score'].max():.3f}]")

    # Step 3: Calculate composite stress score
    print("\nStep 3: Calculate composite stress score")
    detector = StressScoreDetector()
    df = detector.calculate_stress_score(df)
    print(f"✓ Stress score range: [{df['stress_score'].min():.3f}, {df['stress_score'].max():.3f}]")
    print(f"✓ Mean stress score: {df['stress_score'].mean():.3f}")

    # Step 4: Label states
    print("\nStep 4: Label states")
    labeler = StateLabeler()
    df = labeler.label_state(df)

    state_counts = df['state'].value_counts()
    print("✓ State distribution:")
    for state, count in state_counts.items():
        print(f"  - {state}: {count} ({count/len(df)*100:.1f}%)")

    # Step 5: Display output
    print("\nStep 5: Output sample (last 5 candles)")
    print("\nTimestamp                | Close    | Volatility | Stress  | State")
    print("-" * 80)
    for _, row in df.tail(5).iterrows():
        print(f"{row['timestamp']} | ${row['close']:7.2f} | {row['volatility_score']:10.3f} | {row['stress_score']:7.3f} | {row['state']}")

    # Step 6: Save output
    output_path = '/home/ross/.openclaw/workspace/stop-hunt-detector/example_output.csv'
    df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
        'volatility_score', 'liquidity_score', 'continuation_score', 'speed_score',
        'stress_score', 'state']].to_csv(output_path, index=False)

    print(f"\n✓ Output saved to: {output_path}")

    print("\n" + "="*60)
    print("✓ Pipeline completed successfully")
    print("="*60 + "\n")

    return df


if __name__ == '__main__':
    df = main()
