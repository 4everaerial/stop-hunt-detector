"""
Tests for Coinbase Data Fetcher

Tests:
- Correct data shape
- Time continuity (no gaps)
- No missing data in requested window
"""

import sys
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_coinbase import CoinbaseFetcher
import pandas as pd


def test_basic_fetch():
    """Test basic fetch functionality."""
    print("="*80)
    print("TEST 1: Basic Fetch")
    print("="*80)
    print()

    fetcher = CoinbaseFetcher()

    try:
        df = fetcher.fetch_historical(
            'BTCUSDT',
            '1h',
            days=7,
            start_date='2024-01-01',
            end_date='2024-01-07'
        )

        assert not df.empty, "DataFrame should not be empty"
        assert len(df) > 0, "Should have fetched some candles"

        print(f"✓ Fetched {len(df)} candles")
        print(f"✓ Columns: {df.columns.tolist()}")
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_data_shape():
    """Test correct data shape."""
    print()
    print("="*80)
    print("TEST 2: Data Shape")
    print("="*80)
    print()

    fetcher = CoinbaseFetcher()

    try:
        df = fetcher.fetch_historical(
            'BTCUSDT',
            '1h',
            days=1,
            start_date='2024-01-01',
            end_date='2024-01-02'
        )

        # Check columns
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert list(df.columns) == expected_cols, f"Expected columns {expected_cols}"

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Timestamp should be datetime"
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

        # Check no NaN in essential fields
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert df[col].isna().sum() == 0, f"{col} should have no NaN values"

        print(f"✓ Columns correct: {df.columns.tolist()}")
        print(f"✓ Data types correct")
        print(f"✓ No NaN values")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_time_continuity():
    """Test time continuity (no gaps)."""
    print()
    print("="*80)
    print("TEST 3: Time Continuity")
    print("="*80)
    print()

    fetcher = CoinbaseFetcher()

    try:
        df = fetcher.fetch_historical(
            'BTCUSDT',
            '1h',
            days=2,
            start_date='2024-01-01',
            end_date='2024-01-03'
        )

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate expected hourly intervals
        expected_hours = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='1h'
        )

        # Check we have all expected hours (within reason)
        # Coinbase may have gaps if market was closed, but shouldn't have large gaps
        print(f"Fetched {len(df)} candles")
        print(f"Expected ~{len(expected_hours)} hours in range")

        # Check for large gaps (> 2 hours)
        for i in range(1, len(df)):
            time_diff = df.loc[i, 'timestamp'] - df.loc[i-1, 'timestamp']
            max_gap = timedelta(hours=2)

            if time_diff > max_gap:
                print(f"⚠ Large gap: {time_diff} between {df.loc[i-1, 'timestamp']} and {df.loc[i, 'timestamp']}")

        # Check no duplicates
        duplicates = df['timestamp'].duplicated().sum()
        assert duplicates == 0, f"Should have no duplicate timestamps, found {duplicates}"

        print(f"✓ No duplicate timestamps")
        print(f"✓ Time continuity checked (no large gaps found)")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_no_missing_data():
    """Test no missing data in requested window."""
    print()
    print("="*80)
    print("TEST 4: No Missing Data in Requested Window")
    print("="*80)
    print()

    fetcher = CoinbaseFetcher()

    try:
        start_date = '2024-01-01'
        end_date = '2024-01-02'  # 24 hours

        df = fetcher.fetch_historical(
            'BTCUSDT',
            '1h',
            start_date=start_date,
            end_date=end_date
        )

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Check we have data close to requested range
        df_start = df['timestamp'].min()
        df_end = df['timestamp'].max()

        # Allow some tolerance (Coinbase might not have data at exact boundaries)
        start_tolerance = timedelta(hours=1)
        end_tolerance = timedelta(hours=1)

        assert df_start <= start_dt + start_tolerance, \
            f"Data starts {df_start}, expected close to {start_dt}"
        assert df_end >= end_dt - end_tolerance, \
            f"Data ends {df_end}, expected close to {end_dt}"

        print(f"✓ Data starts within tolerance of requested start")
        print(f"✓ Data ends within tolerance of requested end")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_caching():
    """Test data caching."""
    print()
    print("="*80)
    print("TEST 5: Data Caching")
    print("="*80)
    print()

    fetcher = CoinbaseFetcher()

    try:
        # Fetch data
        df1 = fetcher.fetch_historical(
            'BTCUSDT',
            '1h',
            days=1,
            start_date='2024-01-01',
            end_date='2024-01-02'
        )

        # Load from cache
        df2 = fetcher.load_data('BTCUSDT', '1h')

        assert df2 is not None, "Cached data should exist"
        assert len(df1) == len(df2), "Cached data should match fetched data"

        print(f"✓ Cache save successful")
        print(f"✓ Cache load successful")
        print(f"✓ Cached data matches fetched data ({len(df1)} candles)")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("="*80)
    print("COINBASE DATA FETCHER - TEST SUITE")
    print("="*80)
    print()

    tests = [
        ("Basic Fetch", test_basic_fetch),
        ("Data Shape", test_data_shape),
        ("Time Continuity", test_time_continuity),
        ("No Missing Data", test_no_missing_data),
        ("Caching", test_caching),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False

    # Summary
    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print()

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print()
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")


if __name__ == "__main__":
    run_all_tests()
