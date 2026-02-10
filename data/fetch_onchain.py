"""
On-Chain Metrics Fetcher - Coin Metrics Community API (Public)

Fetches stateful on-chain metrics without authentication.
Data source: Coin Metrics Community API (https://api.coinmetrics.io)

This is context data only - NOT used in stress scoring.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlencode


class OnChainFetcher:
    """Fetch public on-chain metrics from Coin Metrics Community API."""

    def __init__(self, base_url="https://api.coinmetrics.io/v4"):
        self.base_url = base_url

    def fetch_metrics(
        self,
        asset="btc",
        metrics=None,
        frequency="1d",
        start_time=None,
        end_time=None,
        save=True
    ):
        """
        Fetch on-chain metrics for a given asset.

        Args:
            asset: Asset symbol (e.g., "btc")
            metrics: List of metrics to fetch
            frequency: Data frequency (e.g., "1d", "1h" if supported)
            start_time: ISO8601 start timestamp
            end_time: ISO8601 end timestamp
            save: Save to cache

        Returns:
            DataFrame with timestamp + metrics columns
        """
        if metrics is None:
            # Minimal, stateful metrics
            metrics = [
                "SplyCur",        # Current supply
                "AdrActCnt",      # Active addresses
                "TxTfrValAdjUSD"  # Adjusted transfer value (USD)
            ]

        endpoint = f"/timeseries/asset-metrics"
        url = f"{self.base_url}{endpoint}"

        params = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": frequency,
        }

        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "data" not in data or not data["data"]:
                return pd.DataFrame()

            df = pd.DataFrame(data["data"])

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["time"], utc=True)
            df = df.drop(columns=["time"], errors="ignore")

            # Convert numeric metrics
            for m in metrics:
                if m in df.columns:
                    df[m] = pd.to_numeric(df[m], errors="coerce")

            # Rename to snake_case columns
            rename_map = {
                "SplyCur": "onchain_supply",
                "AdrActCnt": "onchain_active_addresses",
                "TxTfrValAdjUSD": "onchain_transfer_value_usd",
            }
            df = df.rename(columns=rename_map)

            # Keep only timestamp + metrics
            cols = ["timestamp"] + [rename_map.get(m, m) for m in metrics]
            df = df[cols]

            # Save to cache if requested
            if save:
                cache_path = f"data/historical/{asset}_{frequency}_onchain.csv"
                df.to_csv(cache_path, index=False)
                print(f"✓ Saved on-chain metrics to {cache_path}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching on-chain metrics: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    fetcher = OnChainFetcher()
    print("Fetching BTC on-chain metrics (daily, last 180 days)...")

    # Default is full range if no start/end; limit via start_time
    start_time = (datetime.now(timezone.utc) - pd.Timedelta(days=180)).isoformat()
    df = fetcher.fetch_metrics(asset="btc", frequency="1d", start_time=start_time)

    if not df.empty:
        print(df.head())
    else:
        print("No data fetched")
