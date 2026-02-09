"""
Liquidation Event Tagger

Manually tag historical liquidation events for validation.
These are events where massive forced liquidations occurred.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class EventTagger:
    """Manages liquidation event tagging."""

    EVENTS_FILE = Path(__file__).parent.parent / "events" / "tagged_events.json"

    # Pre-defined major liquidation events (reference)
    KNOWN_EVENTS = {
        "FTX Collapse": {
            "date": "2022-11-08",
            "pair": "BTCUSDT",
            "description": "FTX bankruptcy triggered massive liquidations",
            "source": "news"
        },
        "Luna Collapse": {
            "date": "2022-05-09",
            "pair": "LUNAUSDT",
            "description": "Terra/Luna death spiral",
            "source": "news"
        },
        "COVID Crash": {
            "date": "2020-03-12",
            "pair": "BTCUSDT",
            "description": "COVID-19 market panic, 50% BTC drop",
            "source": "news"
        },
        "2022 Bear Trap": {
            "date": "2022-06-13",
            "pair": "BTCUSDT",
            "description": "Celsius liquidity crisis triggered liquidations",
            "source": "news"
        }
    }

    def __init__(self):
        self.EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.events = self._load_events()

    def _load_events(self) -> List[Dict]:
        """Load events from file, or initialize empty list."""
        if self.EVENTS_FILE.exists():
            with open(self.EVENTS_FILE, 'r') as f:
                return json.load(f)
        return []

    def _save_events(self):
        """Save events to file."""
        with open(self.EVENTS_FILE, 'w') as f:
            json.dump(self.events, f, indent=2)

    def add_event(
        self,
        pair: str,
        timestamp: str,
        description: str,
        source: str = "manual",
        severity: str = "high"
    ):
        """
        Add a liquidation event.

        Args:
            pair: Trading pair (e.g., "BTCUSDT")
            timestamp: ISO timestamp or date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
            description: What happened
            source: "manual", "news", "scan"
            severity: "high", "medium", "low"
        """
        event = {
            "pair": pair.upper(),
            "timestamp": timestamp,
            "description": description,
            "source": source,
            "severity": severity,
            "added_at": datetime.now().isoformat()
        }

        self.events.append(event)
        self._save_events()
        print(f"✓ Added event: {pair} @ {timestamp}")

    def list_events(self, pair: Optional[str] = None) -> List[Dict]:
        """List all events, optionally filtered by pair."""
        if pair:
            return [e for e in self.events if e['pair'].upper() == pair.upper()]
        return self.events

    def remove_event(self, index: int):
        """Remove event by index."""
        if 0 <= index < len(self.events):
            removed = self.events.pop(index)
            self._save_events()
            print(f"✗ Removed event: {removed['pair']} @ {removed['timestamp']}")
        else:
            print("Invalid index")

    def load_known_events(self):
        """Load pre-defined known events as reference."""
        print("\nLoading known liquidation events as reference:")
        for name, event in self.KNOWN_EVENTS.items():
            print(f"  • {name}: {event['date']} - {event['description']}")
            # Don't add automatically - user should review and add manually

    def export_validation_set(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Export events as a DataFrame for validation pipeline.

        Args:
            output_file: Optional CSV file to export to

        Returns:
            DataFrame with events
        """
        import pandas as pd

        df = pd.DataFrame(self.events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Exported to {output_file}")

        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tag liquidation events")
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Add event command
    add_parser = subparsers.add_parser('add', help='Add an event')
    add_parser.add_argument('--pair', required=True, help='Trading pair (e.g., BTCUSDT)')
    add_parser.add_argument('--time', required=True, help='Timestamp (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    add_parser.add_argument('--desc', required=True, help='Description')
    add_parser.add_argument('--severity', default='high', help='Severity (high/medium/low)')

    # List command
    list_parser = subparsers.add_parser('list', help='List events')
    list_parser.add_argument('--pair', help='Filter by pair')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove event')
    remove_parser.add_argument('--index', type=int, required=True, help='Event index')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export validation set')
    export_parser.add_argument('--output', help='Output CSV file')

    # Load known command
    subparsers.add_parser('known', help='List known events for reference')

    args = parser.parse_args()

    tagger = EventTagger()

    if args.command == 'add':
        tagger.add_event(
            pair=args.pair,
            timestamp=args.time,
            description=args.desc,
            severity=args.severity
        )
    elif args.command == 'list':
        events = tagger.list_events(pair=args.pair)
        for i, event in enumerate(events):
            print(f"[{i}] {event['pair']} @ {event['timestamp']} - {event['description']}")
    elif args.command == 'remove':
        tagger.remove_event(args.index)
    elif args.command == 'export':
        tagger.export_validation_set(output_file=args.output)
    elif args.command == 'known':
        tagger.load_known_events()
    else:
        parser.print_help()
