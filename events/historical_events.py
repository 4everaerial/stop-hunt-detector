"""
Historical Liquidation Events

Pre-tagged major liquidation events for validation.
Source: News reports, historical market data, industry publications.
"""

HISTORICAL_EVENTS = [
    {
        "pair": "BTCUSDT",
        "timestamp": "2020-03-12 08:00:00",
        "description": "COVID-19 Market Crash",
        "severity": "high",
        "notes": "50% BTC drop in 24 hours, massive liquidations",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2021-05-19 08:00:00",
        "description": "China Mining Crackdown",
        "severity": "high",
        "notes": "China bans bitcoin mining, 30% drop, cascading liquidations",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2021-11-26 08:00:00",
        "description": "Omicron Variant Panic",
        "severity": "medium",
        "notes": "COVID variant fears, sharp but brief selloff",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2022-01-24 08:00:00",
        "description": "Russia-Ukraine Tensions",
        "severity": "high",
        "notes": "Geopolitical fears, risk-off in crypto",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2022-05-09 08:00:00",
        "description": "Terra/Luna Collapse",
        "severity": "high",
        "notes": "UST de-pegging triggered broad crypto crash, $45B wiped out",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2022-06-13 08:00:00",
        "description": "Celsius Liquidity Crisis",
        "severity": "high",
        "notes": "Major DeFi lender halts withdrawals, contagion fears",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2022-11-08 08:00:00",
        "description": "FTX Bankruptcy",
        "severity": "high",
        "notes": "FTX collapse, $8B frozen, extreme volatility",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2023-03-11 08:00:00",
        "description": "Silvergate Collapse",
        "severity": "medium",
        "notes": "Crypto bank failure, short-lived selloff",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2023-06-05 08:00:00",
        "description": "SEC Suing Binance/Coinbase",
        "severity": "medium",
        "notes": "Regulatory crackdown news, sharp drop",
        "source": "news"
    },
    {
        "pair": "BTCUSDT",
        "timestamp": "2024-03-14 08:00:00",
        "description": "Bitcoin Halving Preceding Volatility",
        "severity": "medium",
        "notes": "Pre-halving volatility, large swings",
        "source": "news"
    },
    {
        "pair": "ETHUSDT",
        "timestamp": "2022-05-09 08:00:00",
        "description": "Terra/Luna Collapse (ETH)",
        "severity": "high",
        "notes": "Broad crypto market crash, ETH dropped 35%",
        "source": "news"
    },
    {
        "pair": "ETHUSDT",
        "timestamp": "2022-06-13 08:00:00",
        "description": "Celsius Crisis (ETH)",
        "severity": "high",
        "notes": "ETH dropped 40% during liquidity crisis",
        "source": "news"
    },
    {
        "pair": "ETHUSDT",
        "timestamp": "2022-11-08 08:00:00",
        "description": "FTX Collapse (ETH)",
        "severity": "high",
        "notes": "ETH dropped 30% in FTX aftermath",
        "source": "news"
    },
]


def get_events_by_pair(pair: str) -> list:
    """Get events filtered by trading pair."""
    return [e for e in HISTORICAL_EVENTS if e['pair'].upper() == pair.upper()]


def get_events_by_severity(severity: str) -> list:
    """Get events filtered by severity."""
    return [e for e in HISTORICAL_EVENTS if e['severity'].lower() == severity.lower()]


def get_events_by_year(year: int) -> list:
    """Get events from a specific year."""
    return [e for e in HISTORICAL_EVENTS if year in e['timestamp']]


def get_all_events() -> list:
    """Get all historical events."""
    return HISTORICAL_EVENTS.copy()


if __name__ == "__main__":
    import json

    print("="*80)
    print("HISTORICAL LIQUIDATION EVENTS")
    print("="*80)
    print()

    print(f"Total events: {len(HISTORICAL_EVENTS)}")
    print()

    # Group by pair
    pairs = {}
    for event in HISTORICAL_EVENTS:
        pair = event['pair']
        if pair not in pairs:
            pairs[pair] = []
        pairs[pair].append(event)

    print("By pair:")
    for pair, events in pairs.items():
        print(f"  {pair}: {len(events)} events")
    print()

    # Group by severity
    severities = {}
    for event in HISTORICAL_EVENTS:
        sev = event['severity']
        if sev not in severities:
            severities[sev] = []
        severities[sev].append(event)

    print("By severity:")
    for sev, events in severities.items():
        print(f"  {sev.upper()}: {len(events)} events")
    print()

    # List all events
    print("All events:")
    for i, event in enumerate(HISTORICAL_EVENTS, 1):
        print(f"  {i}. {event['description']} ({event['timestamp']}) - {event['pair']}")
    print()

    # Save to JSON
    output_path = '/home/ross/.openclaw/workspace/stop-hunt-detector/events/historical_events.json'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(HISTORICAL_EVENTS, f, indent=2)
    print(f"✓ Events saved to {output_path}")
