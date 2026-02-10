"""
Open Interest Fetcher - Binance Futures Public API

Fetches historical open interest for perpetual futures.
Data source: Binance Public API (no auth required).
This is context data only - NOT used in stress scoring.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time


class OpenInterestFetcher:
    """Fetch historical open interest from Binance Futures public API."""

    def __init__(self, base_url="https://fapi.binance.com"):
        self.base_url = base_url

    def fetch_open_interest(self, symbol="BTCUSDT", interval="1h", limit=500, end_time=None):
        """
        Fetch open interest history from Binance Futures.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1h", "4h", "1d")
            limit: Number of data points (max 500)
            end_time: End timestamp in ms (default: now)

        Returns:
            DataFrame with columns: timestamp, open_interest
        """
        endpoint = "/fapi/v1/openInterest"
        url = f"{self.base_url}{endpoint}"

        params = {
            "symbol": symbol,
            "period": interval,
            "limit": limit,
        }

        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse response
            df = pd.DataFrame(data)

            if not df.empty:
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

                # Reorder columns
                df = df[['timestamp', 'openInterest']]
                df.columns = ['timestamp', 'open_interest']

                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching open interest: {e}")
            return pd.DataFrame()

    def fetch_historical(self, symbol="BTCUSDT", interval="1h", hours=720, save=True):
        """
        Fetch historical open interest for a time range.

        Args:
            symbol: Trading pair
            interval: Kline interval
            hours: Number of hours of history to fetch
            save: Whether to save to cache

        Returns:
            DataFrame with timestamp and open_interest
        """
        all_data = []
        current_end = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Calculate iterations (max 500 points per call)
        points_per_call = 500
        calls_needed = (hours + points_per_call - 1) // points_per_call

        for i in range(calls_needed):
            df = self.fetch_open_interest(
                symbol=symbol,
                interval=interval,
                limit=points_per_call,
                end_time=current_end
            )

            if not df.empty:
                all_data.append(df)

            # Move end time back
            if not df.empty:
                current_end = int(df['timestamp'].min().timestamp() * 1000) - 1
            else:
                # Move back by interval * limit if no data
                current_end -= self._interval_to_ms(interval) * points_per_call

            # Rate limit
            time.sleep(0.2)

        # Combine all data
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('timestamp').reset_index(drop=True)

            # Save to cache if requested
            if save:
                cache_path = f"data/historical/{symbol}_{interval}_open_interest.csv"
                result.to_csv(cache_path, index=False)
                print(f"✓ Saved open interest to {cache_path}")

            return result

        return pd.DataFrame()

    def _interval_to_ms(self, interval):
        """Convert interval string to milliseconds."""
        mapping = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return mapping.get(interval, 60 * 60 * 1000)  # Default to 1h


if __name__ == "__main__":
    # Test fetcher
    fetcher = OpenInterestFetcher()

    print("Fetching BTCUSDT 1h open interest (past 24h)...")
    df = fetcher.fetch_open_interest(symbol="BTCUSDT", interval="1h", limit=24)

    if not df.empty:
        print(f"\nFetched {len(df)} records")
        print(df.head())
    else:
        print("No data fetched")
