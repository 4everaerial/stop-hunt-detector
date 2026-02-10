#!/usr/bin/env python3
"""
Day 1 validation script.

Tests:
- Data ingestion pipeline for OHLCV data (Binance API)
- Volatility compression signal (range tightening, ATR decay)
- Basic stress score composition (0-1 normalization)
- Labeled state output (NORMAL/STRESSED/FRAGILE/IMMINENT_CLEARING)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import project modules
from data.fetch_binance import fetch_historical_data
from signals.volatility import VolatilitySignal
from detector.stress_score import StressScoreDetector
from detector.state_label import StateLabeler


def test_data_ingestion():
    """Test data ingestion from Binance API."""
    print("\n=== Test 1: Data Ingestion ===")

    try:
        # Fetch 30 days of BTC/USDT 1h data (small sample for testing)
        print("Fetching 30 days of BTCUSDT 1h data from Binance...")
        df = fetch_historical_data('BTCUSDT', '1h', days=30)

        # Validate data
        assert len(df) > 0, "No data fetched"
        assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']), \
            "Missing required columns"
        assert df['timestamp'].is_monotonic_increasing, "Timestamps not sorted"

        print(f"✓ Fetched {len(df)} candles")
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"✓ Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_volatility_signal(df):
    """Test volatility compression signal."""
    print("\n=== Test 2: Volatility Signal ===")

    try:
        # Calculate volatility signal
        print("Calculating volatility compression signal...")
        signal = VolatilitySignal()
        df = signal.calculate(df)

        # Validate output
        assert 'volatility_score' in df.columns, "Missing volatility_score column"
        assert df['volatility_score'].notna().any(), "All volatility scores are NaN"
        assert df['volatility_score'].min() >= 0.0, "Volatility score below 0"
        assert df['volatility_score'].max() <= 1.0, "Volatility score above 1"

        print(f"✓ Volatility score range: [{df['volatility_score'].min():.3f}, {df['volatility_score'].max():.3f}]")
        print(f"✓ Mean volatility score: {df['volatility_score'].mean():.3f}")
        print(f"✓ Signal columns: {[col for col in df.columns if 'volatility' in col.lower() or 'atr' in col.lower() or 'bb' in col.lower()]}")

        return df

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_stress_score_composition(df):
    """Test stress score composition."""
    print("\n=== Test 3: Stress Score Composition ===")

    try:
        # Initialize placeholder scores for non-volatility signals (Day 1 only has volatility)
        df['liquidity_score'] = 0.0
        df['continuation_score'] = 0.0
        df['speed_score'] = 0.0

        # Calculate composite stress score
        print("Calculating composite stress score...")
        detector = StressScoreDetector()
        df = detector.calculate_stress_score(df)

        # Validate output
        assert 'stress_score' in df.columns, "Missing stress_score column"
        assert df['stress_score'].notna().any(), "All stress scores are NaN"
        assert df['stress_score'].min() >= 0.0, "Stress score below 0"
        assert df['stress_score'].max() <= 1.0, "Stress score above 1"

        print(f"✓ Stress score range: [{df['stress_score'].min():.3f}, {df['stress_score'].max():.3f}]")
        print(f"✓ Mean stress score: {df['stress_score'].mean():.3f}")
        print(f"✓ Default weights: {detector.weights}")

        return df

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_state_labeling(df):
    """Test state labeling."""
    print("\n=== Test 4: State Labeling ===")

    try:
        # Add state labels
        print("Labeling states based on stress score...")
        labeler = StateLabeler()
        df = labeler.label_state(df)

        # Validate output
        assert 'state' in df.columns, "Missing state column"
        assert set(df['state'].unique()).issubset({'NORMAL', 'STRESSED', 'FRAGILE', 'IMMINENT_CLEARING', 'UNKNOWN'}), \
            "Invalid state labels"

        # Count states
        state_counts = df['state'].value_counts()
        print(f"✓ State distribution:")
        for state, count in state_counts.items():
            print(f"  - {state}: {count} ({count/len(df)*100:.1f}%)")

        # Validate threshold logic
        normal_df = df[df['state'] == 'NORMAL']
        if len(normal_df) > 0:
            assert normal_df['stress_score'].max() <= 0.3, "NORMAL state threshold violated"

        imminent_df = df[df['state'] == 'IMMINENT_CLEARING']
        if len(imminent_df) > 0:
            assert imminent_df['stress_score'].min() >= 0.8, "IMMINENT_CLEARING state threshold violated"

        return df

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_full_pipeline():
    """Run full Day 1 pipeline test."""
    print("\n" + "="*60)
    print("DAY 1 VALIDATION: Full Pipeline Test")
    print("="*60)

    try:
        # Run all tests
        df = test_data_ingestion()
        df = test_volatility_signal(df)
        df = test_stress_score_composition(df)
        df = test_state_labeling(df)

        # Save sample output
        output_path = '/home/ross/.openclaw/workspace/stop-hunt-detector/test_output_day1.csv'
        df[['timestamp', 'close', 'volatility_score', 'stress_score', 'state']].to_csv(output_path, index=False)
        print(f"\n✓ Sample output saved to: {output_path}")

        # Display last 5 rows
        print("\n=== Last 5 Candles ===")
        print(df[['timestamp', 'close', 'volatility_score', 'stress_score', 'state']].tail().to_string(index=False))

        print("\n" + "="*60)
        print("✓ DAY 1 VALIDATION PASSED")
        print("="*60)

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("✗ DAY 1 VALIDATION FAILED")
        print("="*60)
        print(f"Error: {e}")
        return False


if __name__ == '__main__':
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
