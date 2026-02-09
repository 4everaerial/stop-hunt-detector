"""
Mock Data Generator

Generates synthetic OHLCV data for testing when real data is unavailable.
Note: For production use with real Binance data, ensure proper access
(possibly via VPN or proxy due to Binance geoblocking).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_data(
    start_date: str = '2024-01-01',
    days: int = 90,
    base_price: float = 60000,
    volatility: float = 0.02,
    trend: float = 0.0005,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        start_date: Start date (YYYY-MM-DD)
        days: Number of days to generate
        base_price: Starting price
        volatility: Daily volatility
        trend: Daily price trend (positive = up, negative = down)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    # Generate hourly candles
    hours_per_day = 24
    total_candles = days * hours_per_day

    # Generate price path with trend and volatility
    drift = trend / hours_per_day
    vol_per_hour = volatility / np.sqrt(hours_per_day)

    returns = np.random.normal(drift, vol_per_hour, total_candles)
    prices = base_price * np.cumprod(1 + returns)

    # Add some volatility compression/expansion cycles
    cycle_length = 24  # 1 day cycle
    for i in range(len(returns)):
        cycle_pos = i % cycle_length
        # Compress volatility at start of cycle, expand at end
        vol_factor = 0.5 + 1.5 * (cycle_pos / cycle_length) ** 2
        prices[i] = base_price * (1 + returns[i] * vol_factor)

    # Generate OHLCV
    timestamps = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(hours=i) for i in range(total_candles)]

    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Generate OHLC from close with some noise
        candle_range = close_price * vol_per_hour * (0.5 + np.random.random())

        open_price = close_price * (1 - np.random.normal(0, vol_per_hour))
        open_price = max(open_price, 100)  # Minimum price

        high_price = max(open_price, close_price) + candle_range * np.random.random()
        low_price = min(open_price, close_price) - candle_range * np.random.random()

        # Volume (correlated with volatility)
        vol_base = 1000 + (candle_range / close_price) * 100000
        volume = vol_base * (0.5 + np.random.random())

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


def generate_stress_event_data(
    start_date: str = '2024-01-15',
    event_day: int = 30,
    days: int = 60,
    base_price: float = 60000
) -> pd.DataFrame:
    """
    Generate data with a simulated stress event (stop hunt scenario).

    The stress event has:
    - Volatility compression before event
    - Sharp price drop (stop hunt)
    - Recovery after event

    Args:
        start_date: Start date
        event_day: Day when event occurs (from start)
        days: Total days
        base_price: Starting price

    Returns:
        DataFrame with synthetic data containing stress event
    """
    df = generate_mock_data(start_date, days, base_price, seed=123)

    # Modify data around event to create stress patterns
    event_hour = event_day * 24

    # 1. Volatility compression (24-48 hours before event)
    compression_start = event_hour - 48
    compression_end = event_hour - 24

    for i in range(compression_start, compression_end):
        if 0 <= i < len(df):
            # Tighten the range
            center_price = df.iloc[i]['close']
            df.at[i, 'high'] = center_price * 1.005
            df.at[i, 'low'] = center_price * 0.995
            # Reduce volume
            df.at[i, 'volume'] *= 0.5

    # 2. Stop hunt event (sharp drop)
    for i in range(event_hour, min(event_hour + 6, len(df))):
        if i < len(df):
            # Sharp drop
            drop_factor = 1.0 - 0.03 * (i - event_hour + 1) / 6
            df.at[i, 'close'] *= drop_factor
            df.at[i, 'low'] *= drop_factor
            # Large wick
            df.at[i, 'low'] *= 0.98
            # High volume
            df.at[i, 'volume'] *= 3.0
            df.at[i, 'open'] = df.at[i-1, 'close'] if i > 0 else df.at[i, 'close'] * 1.001

    # 3. Recovery
    for i in range(event_hour + 6, min(event_hour + 24, len(df))):
        if i < len(df):
            recovery_factor = 1.0 + 0.01 * (i - event_hour - 6) / 18
            df.at[i, 'close'] *= recovery_factor
            df.at[i, 'high'] *= recovery_factor

    return df


if __name__ == "__main__":
    # Generate some test data
    print("Generating mock data with stress event...")

    df = generate_stress_event_data(
        start_date='2024-01-01',
        event_day=30,
        days=60
    )

    print(f"Generated {len(df)} candles")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nPrice range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # Save to file
    output_path = '/home/ross/.openclaw/workspace/stop-hunt-detector/data/historical/BTCUSDT_1h_mock.csv'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
