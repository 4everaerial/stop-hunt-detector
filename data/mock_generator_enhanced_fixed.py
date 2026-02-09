"""
Mock Data Generator (Enhanced - Fixed)

Generates synthetic OHLCV data with realistic stress event signatures.
Fixed version that ensures all prices are non-zero and valid.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_enhanced_stress_event_data(
    start_date: str = '2024-01-15',
    event_day: int = 30,
    days: int = 90,
    base_price: float = 60000
) -> pd.DataFrame:
    """
    Generate data with realistic stress event patterns.

    Creates:
    1. Calm baseline period (days 1-20)
    2. Volatility compression (days 21-28) - range tightens
    3. Stop hunt event (day 30) - sharp spike down then reversal
    4. Recovery period (days 31-45)
    5. Return to calm (days 46-90)

    Args:
        start_date: Start date
        event_day: Day when event occurs
        days: Total days
        base_price: Starting price

    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(123)
    hours_per_day = 24
    total_candles = days * hours_per_day

    # Phase definitions (in candle indices)
    baseline_end = 20 * hours_per_day
    compression_start = baseline_end
    compression_end = 28 * hours_per_day
    event_start = (event_day - 1) * hours_per_day
    event_end = event_start + 6  # 6-hour event
    recovery_end = 45 * hours_per_day

    # Generate timestamps
    timestamps = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(hours=i) for i in range(total_candles)]

    # Initialize arrays with base price (not zeros!)
    prices = np.full(total_candles, base_price)
    volumes = np.full(total_candles, 1000.0)

    # Phase 1: Baseline (calm)
    for i in range(0, baseline_end):
        if i == 0:
            prices[i] = base_price
        else:
            # Low volatility random walk
            change = np.random.normal(0.0002, 0.003)  # 0.3% hourly vol
            prices[i] = prices[i-1] * (1 + change)
            # Ensure price stays positive
            prices[i] = max(prices[i], 1000)
        volumes[i] = 1000 + np.random.normal(0, 200)

    # Phase 2: Compression (range tightens)
    for i in range(compression_start, min(compression_end, total_candles)):
        # Decreasing volatility
        progress = (i - compression_start) / (compression_end - compression_start)
        vol = 0.003 * (1 - 0.6 * progress)  # Vol drops to 40% of baseline

        change = np.random.normal(0.0001, vol)
        prices[i] = prices[i-1] * (1 + change)
        prices[i] = max(prices[i], 1000)

        # Lower volume during compression
        volumes[i] = 600 + np.random.normal(0, 100)

    # Phase 3: Stop hunt event (sharp spike)
    for i in range(event_start, min(event_end, total_candles)):
        event_progress = (i - event_start) / (event_end - event_start)

        if i == event_start:
            prices[i] = prices[i-1] * 1.002  # Fake breakout
        elif event_progress < 0.3:
            # Sharp drop
            prices[i] = prices[i-1] * 0.985  # 1.5% drop per candle
        elif event_progress < 0.7:
            # Cap out, reverse
            prices[i] = prices[i-1] * 1.025  # 2.5% recovery
        else:
            # Recovery continues
            prices[i] = prices[i-1] * 1.008

        prices[i] = max(prices[i], 1000)

        # Spike volume
        volumes[i] = 5000 + np.random.normal(0, 500)

    # Phase 4: Recovery and return to normal
    for i in range(event_end, total_candles):
        if i < recovery_end:
            # Gradual return to normal
            vol = 0.003 + 0.002 * (1 - (recovery_end - i) / (recovery_end - event_end))
        else:
            vol = 0.003

        change = np.random.normal(0.0002, vol)
        prices[i] = prices[i-1] * (1 + change)
        prices[i] = max(prices[i], 1000)

        if i < recovery_end + 24:
            volumes[i] = 2000 + np.random.normal(0, 300)
        else:
            volumes[i] = 1000 + np.random.normal(0, 200)

    # Fill any remaining gaps
    for i in range(total_candles):
        if prices[i] <= 0:
            prices[i] = max(prices[i-1] if i > 0 else base_price, 1000)
        if volumes[i] <= 0:
            volumes[i] = 1000

    # Generate OHLC from price path
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Add candle-to-candle noise
        candle_vol = 0.001 + 0.002 * abs(np.random.randn())

        open_price = close_price * (1 - np.random.normal(0, candle_vol))
        open_price = max(open_price, 1000)  # Ensure positive

        # Generate realistic OHLC structure
        direction = 1 if close_price >= open_price else -1
        body = abs(close_price - open_price)
        upper_wick = body * (0.1 + 0.8 * np.random.random()) * (0.5 + 0.5 * np.random.random())
        lower_wick = body * (0.1 + 0.8 * np.random.random()) * (0.5 + 0.5 * np.random.random())

        if direction == 1:  # Green candle
            high_price = max(open_price, close_price) + upper_wick
            low_price = min(open_price, close_price) - lower_wick
        else:  # Red candle
            high_price = max(open_price, close_price) + upper_wick
            low_price = min(open_price, close_price) - lower_wick

        # Ensure valid OHLC
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        low_price = max(low_price, 100)

        # Normalize volume
        volume = max(volumes[i], 100)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Generate enhanced test data
    print("Generating enhanced stress event data...")

    df = generate_enhanced_stress_event_data(
        start_date='2024-01-01',
        event_day=30,
        days=90
    )

    print(f"Generated {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print()

    # Check for issues
    print("Data validation:")
    print(f"  Any zeros: {(df == 0).any().any()}")
    print(f"  Any negative prices: {(df[['open', 'high', 'low', 'close']] < 0).any().any()}")
    print(f"  Any NaN: {df.isna().any().any()}")

    # Save to file
    output_path = '/home/ross/.openclaw/workspace/stop-hunt-detector/data/historical/BTCUSDT_1h_enhanced.csv'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
