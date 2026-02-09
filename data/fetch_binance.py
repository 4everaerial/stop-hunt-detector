"""
Binance OHLCV Data Fetcher

Fetches historical candlestick (OHLCV) data from Binance public API.
Supports multiple pairs, timeframes, and date ranges.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import time


class BinanceFetcher:
    """Fetches OHLCV data from Binance public API."""

    BASE_URL = "https://api.binance.com/api/v3/klines"
    DATA_DIR = Path(__file__).parent.parent / "data" / "historical"

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_historical(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            days: Number of days to fetch (if start_date not provided)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format, defaults to now)
            save: Whether to save data to disk

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Calculate date range
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        if start_date:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

        # Fetch data in chunks (Binance limits to 1000 candles per request)
        all_candles = []
        current_start = start_ts

        print(f"Fetching {symbol} {interval} data from {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}")

        while current_start < end_ts:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                candles = response.json()

                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next chunk
                last_timestamp = candles[-1][0]
                current_start = last_timestamp + 1

                # Rate limiting - Binance allows 1200 requests/minute
                time.sleep(0.05)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break

        if not all_candles:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Fetched {len(df)} candles")

        # Save to disk
        if save:
            self.save_data(df, symbol, interval)

        return df

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save DataFrame to CSV."""
        filename = f"{symbol}_{interval}.csv"
        filepath = self.DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")

    def load_data(self, symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Load previously saved OHLCV data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe

        Returns:
            DataFrame with OHLCV data, or None if file doesn't exist
        """
        filename = f"{symbol}_{interval}.csv"
        filepath = self.DATA_DIR / filename

        if not filepath.exists():
            return None

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def fetch_multiple_pairs(
        self,
        pairs: List[str],
        interval: str = "1h",
        days: int = 365
    ) -> dict:
        """
        Fetch data for multiple pairs.

        Args:
            pairs: List of trading pairs
            interval: Timeframe
            days: Number of days

        Returns:
            Dictionary mapping pair -> DataFrame
        """
        data = {}
        for pair in pairs:
            print(f"\n--- Fetching {pair} ---")
            df = self.fetch_historical(pair, interval, days, save=True)
            if not df.empty:
                data[pair] = df
        return data


if __name__ == "__main__":
    # Example usage
    fetcher = BinanceFetcher()

    # Fetch BTC/USDT 1-hour data for the last year
    df = fetcher.fetch_historical('BTCUSDT', '1h', days=365)

    print(f"\n{df.head()}")
    print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df)}")
