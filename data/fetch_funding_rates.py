"""
Funding Rates Fetcher - Binance Futures Public API

Fetches historical funding rates for perpetual futures.
Data source: Binance Public API (no auth required).
This is context data only - NOT used in stress scoring.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time


class FundingRatesFetcher:
    """Fetch historical funding rates from Binance Futures public API."""

    def __init__(self, base_url="https://fapi.binance.com"):
        self.base_url = base_url

    def fetch_funding_rate(self, symbol="BTCUSDT", limit=500, end_time=None):
        """
        Fetch funding rate history from Binance Futures.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Number of data points (max 1000)
            end_time: End timestamp in ms (default: now)

        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        endpoint = "/fapi/v1/fundingRate"
        url = f"{self.base_url}{endpoint}"

        params = {
            "symbol": symbol,
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
                df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
                df['fundingRate'] = df['fundingRate'].astype(float)

                # Reorder columns
                df = df[['timestamp', 'fundingRate']]
                df.columns = ['timestamp', 'funding_rate']

                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching funding rates: {e}")
            return pd.DataFrame()

    def fetch_historical(self, symbol="BTCUSDT", hours=720, save=True):
        """
        Fetch historical funding rates for a time range.

        Args:
            symbol: Trading pair
            hours: Number of hours of history to fetch
            save: Whether to save to cache

        Returns:
            DataFrame with timestamp and funding_rate
        """
        all_data = []
        current_end = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Binance funding rate is 8 hours, so we can fetch up to 1000 records
        # 1000 records * 8 hours = 8000 hours = 333 days
        calls_needed = (hours + 8000 - 1) // 8000
        if calls_needed == 0:
            calls_needed = 1

        for i in range(calls_needed):
            df = self.fetch_funding_rate(
                symbol=symbol,
                limit=1000,
                end_time=current_end
            )

            if not df.empty:
                all_data.append(df)

                # Move end time back (funding rate is every 8 hours = 28800000 ms)
                current_end = int(df['timestamp'].min().timestamp() * 1000) - 1

            # Rate limit
            time.sleep(0.2)

        # Combine all data
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('timestamp').reset_index(drop=True)

            # Save to cache if requested
            if save:
                cache_path = f"data/historical/{symbol}_funding_rate.csv"
                result.to_csv(cache_path, index=False)
                print(f"✓ Saved funding rates to {cache_path}")

            return result

        return pd.DataFrame()


if __name__ == "__main__":
    # Test fetcher
    fetcher = FundingRatesFetcher()

    print("Fetching BTCUSDT funding rates (past 100 records)...")
    df = fetcher.fetch_funding_rate(symbol="BTCUSDT", limit=100)

    if not df.empty:
        print(f"\nFetched {len(df)} records")
        print(df.head())
    else:
        print("No data fetched")
