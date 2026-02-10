"""
Context Snapshot - Data Product, Not a Model

Aligns timestamps across multiple data streams to produce
a combined view: OHLCV, stress, open_interest, funding_rate.

This is a data product for situational awareness.
It does NOT aggregate, score, or predict.
"""

import pandas as pd
from pathlib import Path
from datetime import timezone


class ContextSnapshot:
    """
    Align multiple data streams into a unified timestamped snapshot.

    This is a data product for situational awareness.
    No aggregation, no scoring, no interpretation.
    """

    def __init__(self):
        self.data = {}

    def add_ohlcv(self, df: pd.DataFrame):
        """
        Add OHLCV data (from Coinbase or other source).

        Expected columns: timestamp, open, high, low, close, volume
        """
        self.data['ohlcv'] = df.copy()

    def add_fast_stress(self, df: pd.DataFrame):
        """
        Add fast stress data (from rolling_stress_score).

        Expected columns: timestamp, fast_stress, [stress_state]
        """
        self.data['fast_stress'] = df.copy()

    def add_open_interest(self, df: pd.DataFrame):
        """
        Add open interest data.

        Expected columns: timestamp, open_interest
        """
        self.data['open_interest'] = df.copy()

    def add_funding_rates(self, df: pd.DataFrame):
        """
        Add funding rate data.

        Expected columns: timestamp, funding_rate
        """
        self.data['funding_rates'] = df.copy()

    def add_onchain(self, df: pd.DataFrame):
        """
        Add on-chain metrics.

        Expected columns: timestamp, [metrics...]
        """
        self.data['onchain'] = df.copy()

    def align(self, how='outer', freq='1h'):
        """
        Align all data streams on a common timestamp index.

        Args:
            how: Merge method ('inner' or 'outer')
            freq: Resampling frequency (default '1h')

        Returns:
            DataFrame with aligned columns
        """
        if not self.data:
            return pd.DataFrame()

        # Start with OHLCV if available, otherwise use first available stream
        if 'ohlcv' in self.data:
            base_df = self.data['ohlcv'].copy()
            base_timestamps = base_df['timestamp']
        else:
            # Find a stream with timestamp
            for key, df in self.data.items():
                if 'timestamp' in df.columns:
                    base_df = df[['timestamp']].copy()
                    base_timestamps = base_df['timestamp']
                    break
            else:
                return pd.DataFrame()

        # Align each stream
        aligned_data = {'timestamp': base_timestamps}

        for key, df in self.data.items():
            if 'timestamp' not in df.columns:
                continue

            df_aligned = df.set_index('timestamp')

            # Reindex to base timestamps
            df_aligned = df_aligned.reindex(base_timestamps, method='nearest')

            # Add to aligned data (flatten multi-index columns)
            for col in df_aligned.columns:
                aligned_data[f"{key}_{col}"] = df_aligned[col].values

        # Create aligned DataFrame
        result = pd.DataFrame(aligned_data)

        return result

    def save(self, output_path: str):
        """
        Save aligned snapshot to CSV.

        Args:
            output_path: Path to output file
        """
        aligned_df = self.align()

        if not aligned_df.empty:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            aligned_df.to_csv(output_path, index=False)
            print(f"✓ Saved context snapshot to {output_path}")

            return aligned_df
        else:
            print("No data to save")
            return pd.DataFrame()


def create_snapshot(
    ohlcv_path=None,
    stress_path=None,
    open_interest_path=None,
    funding_rates_path=None,
    output_path="output/context_snapshot.csv"
):
    """
    Convenience function to create a context snapshot from cached files.

    Args:
        ohlcv_path: Path to OHLCV CSV
        stress_path: Path to stress score CSV
        open_interest_path: Path to open interest CSV
        funding_rates_path: Path to funding rates CSV
        output_path: Output path for snapshot

    Returns:
        Aligned DataFrame
    """
    snapshot = ContextSnapshot()

    # Load OHLCV
    if ohlcv_path and Path(ohlcv_path).exists():
        df = pd.read_csv(ohlcv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        snapshot.add_ohlcv(df)
        print(f"✓ Loaded OHLCV: {len(df)} records")

    # Load stress scores
    if stress_path and Path(stress_path).exists():
        df = pd.read_csv(stress_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        snapshot.add_fast_stress(df)
        print(f"✓ Loaded stress: {len(df)} records")

    # Load open interest
    if open_interest_path and Path(open_interest_path).exists():
        df = pd.read_csv(open_interest_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        snapshot.add_open_interest(df)
        print(f"✓ Loaded open interest: {len(df)} records")

    # Load funding rates
    if funding_rates_path and Path(funding_rates_path).exists():
        df = pd.read_csv(funding_rates_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        snapshot.add_funding_rates(df)
        print(f"✓ Loaded funding rates: {len(df)} records")

    # Align and save
    return snapshot.save(output_path)


if __name__ == "__main__":
    # Test snapshot creation
    print("Creating context snapshot from cached data...")

    snapshot_df = create_snapshot(
        ohlcv_path="data/historical/BTCUSDT_1h_coinbase.csv",
        stress_path="output/final_adjudication/scores_timeseries.csv",
        output_path="output/context_snapshot.csv"
    )

    if not snapshot_df.empty:
        print(f"\nCreated snapshot with {len(snapshot_df)} records, {len(snapshot_df.columns)} columns")
        print("\nColumn names:")
        for col in snapshot_df.columns:
            print(f"  - {col}")
    else:
        print("No snapshot created")
