"""
CoinGecko OHLCV Data Fetcher

Alternative data source when Binance is geoblocked.
CoinGecko provides free public API with OHLCV data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import time


class CoinGeckoFetcher:
    """Fetches OHLCV data from CoinGecko public API."""

    BASE_URL = "https://api.coingecko.com/api/v3"
    DATA_DIR = Path(__file__).parent.parent / "data" / "historical"

    # CoinGecko coin IDs
    COIN_IDS = {
        'BTCUSDT': 'bitcoin',
        'ETHUSDT': 'ethereum',
        'SOLUSDT': 'solana',
        'DOGEUSDT': 'dogecoin',
        'ADAUSDT': 'cardano',
        'XRPUSDT': 'ripple',
        'BNBUSDT': 'binancecoin',
        'MATICUSDT': 'matic-network',
    }

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_historical(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 365,
        start_date: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe ("1h", "4h", "1d")
            days: Number of days to fetch
            start_date: Start date (YYYY-MM-DD) - if provided, overrides days
            save: Whether to save to disk

        Returns:
            DataFrame with OHLCV data
        """
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            print(f"❌ Unsupported symbol: {symbol}")
            return pd.DataFrame()

        # CoinGecko uses timestamps in seconds, but we need date ranges
        # For simplicity, fetch the last N days
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            now = datetime.now()
            days_diff = (now - start_dt).days
            days_to_fetch = min(days_diff, 365)  # Max 365 days for free tier
        else:
            days_to_fetch = min(days, 365)

        print(f"Fetching {symbol} {interval} data for {days_to_fetch} days...")

        # CoinGecko OHLCV endpoint
        url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"

        params = {
            'vs_currency': 'usd',
            'days': days_to_fetch
        }

        # Map interval to CoinGecko format
        interval_map = {
            '1h': 'hourly',
            '4h': 'hourly',  # CoinGecko doesn't support 4h, we'll resample
            '1d': 'daily'
        }

        if interval == '1d':
            params['days'] = days_to_fetch
        elif interval == '1h' or interval == '4h':
            # CoinGecko free tier limits hourly data to 90 days
            params['days'] = min(days_to_fetch, 90)
        else:
            print(f"❌ Unsupported interval: {interval}")
            return pd.DataFrame()

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                print(f"❌ No data returned for {symbol}")
                return pd.DataFrame()

            # CoinGecko returns: [timestamp_ms, open, high, low, close]
            # Volume is not included in OHLCV endpoint, need separate call
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Set volume to 0 (CoinGecko OHLCV doesn't include volume)
            df['volume'] = 0

            # Resample if needed
            if interval == '4h':
                df = df.set_index('timestamp').resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()

            print(f"✓ Fetched {len(df)} candles")

            # Save to disk
            if save:
                self.save_data(df, symbol, interval)

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol."""
        return self.COIN_IDS.get(symbol.upper())

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save DataFrame to CSV."""
        filename = f"{symbol}_{interval}_coingecko.csv"
        filepath = self.DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")

    def load_data(self, symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
        """Load previously saved CoinGecko data."""
        filename = f"{symbol}_{interval}_coingecko.csv"
        filepath = self.DATA_DIR / filename

        if not filepath.exists():
            return None

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def fetch_multiple_pairs(
        self,
        pairs: list,
        interval: str = "1h",
        days: int = 365
    ) -> dict:
        """Fetch data for multiple pairs."""
        data = {}
        for pair in pairs:
            print(f"\n--- Fetching {pair} ---")
            df = self.fetch_historical(pair, interval, days, save=True)
            if not df.empty:
                data[pair] = df
                time.sleep(1)  # Rate limiting
        return data


if __name__ == "__main__":
    # Example usage
    fetcher = CoinGeckoFetcher()

    # Fetch BTC/USDT hourly data
    df = fetcher.fetch_historical('BTCUSDT', '1h', days=90)

    print(f"\n{df.head()}")
    print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df)}")
