"""
Coinbase OHLCV Data Fetcher

Fetches historical candlestick (OHLCV) data from Coinbase Exchange API.
Supports arbitrary time ranges via pagination.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List
import time
import configparser


class CoinbaseFetcher:
    """Fetches OHLCV data from Coinbase Exchange API."""

    BASE_URL = "https://api.exchange.coinbase.com"
    DATA_DIR = Path(__file__).parent.parent / "data" / "historical"
    CONFIG_FILE = Path(__file__).parent.parent / ".coinbase_api_key"

    # Supported products
    PRODUCTS = {
        'BTCUSDT': 'BTC-USD',
        'ETHUSDT': 'ETH-USD',
        'SOLUSDT': 'SOL-USD',
        'DOGEUSDT': 'DOGE-USD',
        'ADAUSDT': 'ADA-USD',
        'XRPUSDT': 'XRP-USD',
        'BNBUSDT': 'BNB-USD',
        'MATICUSDT': 'MATIC-USD',
    }

    # Granularity mapping (seconds)
    GRANULARITY = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '4h': 21600,
        '1d': 86400,
    }

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize Coinbase fetcher.

        Args:
            config_file: Optional path to API key config file (unused for public candles)
        """
        if config_file:
            self.CONFIG_FILE = Path(config_file)

        # Public candles endpoint does not require auth
        self.api_key = self._load_api_key(optional=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_api_key(self, optional: bool = False) -> Optional[str]:
        """Load API key from config file (optional for public endpoints)."""
        if not self.CONFIG_FILE.exists():
            return None if optional else None

        config = configparser.ConfigParser()
        config.read(self.CONFIG_FILE)

        if 'coinbase' not in config:
            return None if optional else None

        api_key = config['coinbase'].get('api_key', '')
        api_secret = config['coinbase'].get('api_secret', '')

        if not api_key or not api_secret:
            return None if optional else None

        # Note: Coinbase Exchange public candles do not require auth.
        # Auth handling intentionally omitted.
        return f"{api_key}:{api_secret}"

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
        Fetch historical OHLCV data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            days: Number of days to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save to disk

        Returns:
            DataFrame with OHLCV data
        """
        # Map symbol to Coinbase product
        product_id = self._map_symbol(symbol)
        if not product_id:
            raise ValueError(f"Unsupported symbol: {symbol}")

        # Convert interval to granularity
        granularity = self.GRANULARITY.get(interval)
        if not granularity:
            raise ValueError(f"Unsupported interval: {interval}")

        # Calculate time range
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)

        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=days)

        print(f"Fetching {symbol} {interval} data from {start_dt} to {end_dt}")

        # Fetch with pagination (Coinbase max 300 candles per request)
        all_candles = []

        # Coinbase returns candles in reverse chronological order
        # Paginate backwards from end_dt to start_dt
        page_seconds = granularity * 300
        current_end = end_dt

        while current_end > start_dt:
            page_start = max(start_dt, current_end - timedelta(seconds=page_seconds))

            candles = self._fetch_page(
                product_id,
                granularity,
                page_start,
                current_end
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Move end backward by 1 second to avoid overlap
            current_end = page_start - timedelta(seconds=1)

            # Rate limiting
            time.sleep(0.1)

        if not all_candles:
            print(f"❌ No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'low', 'high', 'open', 'close', 'volume'
        ])

        # Convert timestamp (Unix epoch in milliseconds or seconds)
        # Coinbase API returns Unix timestamp (seconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Reorder to match expected format
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Sort and deduplicate
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

        print(f"✓ Fetched {len(df)} candles")

        # Save to disk
        if save:
            self.save_data(df, symbol, interval)

        return df

    def _fetch_page(
        self,
        product_id: str,
        granularity: int,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[List]:
        """Fetch a single page of candles (max 300)."""
        url = f"{self.BASE_URL}/products/{product_id}/candles"

        params = {
            'granularity': granularity,
            'start': start_dt.isoformat().replace('+00:00', 'Z'),
            'end': end_dt.isoformat().replace('+00:00', 'Z')
        }

        headers = {}

        # Retry loop for rate limits / transient errors
        retries = 5
        backoff = 0.5

        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                if status in (429, 502, 503, 504):
                    sleep_time = backoff * (2 ** attempt)
                    print(f"⚠ Rate limit/transient error ({status}), retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                elif status == 400:
                    # Bad request (often too many candles); return empty
                    print(f"⚠ Bad request (400): {e}")
                    return []
                else:
                    raise
            except requests.exceptions.RequestException as e:
                sleep_time = backoff * (2 ** attempt)
                print(f"⚠ Request error: {e}, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        return []

    def _map_symbol(self, symbol: str) -> Optional[str]:
        """Map trading pair symbol to Coinbase product ID."""
        return self.PRODUCTS.get(symbol.upper())

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save DataFrame to CSV."""
        filename = f"{symbol}_{interval}_coinbase.csv"
        filepath = self.DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")

    def load_data(self, symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
        """Load previously saved Coinbase data."""
        filename = f"{symbol}_{interval}_coinbase.csv"
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
        """Fetch data for multiple pairs."""
        data = {}
        for pair in pairs:
            print(f"\n--- Fetching {pair} ---")
            df = self.fetch_historical(pair, interval, days, save=True)
            if not df.empty:
                data[pair] = df
        return data


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        fetcher = CoinbaseFetcher(config_file=config_file)
    else:
        fetcher = CoinbaseFetcher()

    # Fetch BTC/USDT hourly data
    df = fetcher.fetch_historical(
        'BTCUSDT',
        '1h',
        days=365,
        start_date='2023-01-01',
        end_date='2024-12-31'
    )

    print(f"\n{df.head()}")
    print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
