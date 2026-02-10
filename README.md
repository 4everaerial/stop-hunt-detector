# Market Fragility Estimator — Structural Stress Detection System

**Status:** ✅ Core Engine Stable  
**Architecture:** Fast relative stress detector + orthogonal context streams  
**Repository:** https://github.com/4everaerial/stop-hunt-detector

---

## Overview

A Python-based market fragility estimator that detects **structural instability** in cryptocurrency markets. It measures stress conditions that often precede forced liquidations and market dislocations.

**This is NOT a trading bot.** It does not generate buy/sell signals, predict price direction, or execute trades. It is designed for **situational awareness** and **risk management**.

---

## Core Finding

After extensive validation (real Coinbase BTC-USD data, 2023-2026):

✅ **Fast relative stress detector is anticipatory, sparse, and non-saturating**  
❌ **Slow context separation via OHLCV alone does NOT exist** and has been retired

The dual-scale slow context hypothesis failed: liquidation events do NOT distribute across COLD/NEUTRAL/HOT regimes when using only OHLCV data. All evaluated events occurred in the COLD regime, indicating that slow context from price action alone adds no value.

---

## What It Does

### Detects
- **Volatility Compression:** Tight ranges followed by explosive expansion
- **Liquidity Fragility:** Order book thinning, large wicks on small volume
- **Continuation Failure:** Momentum indicators diverge from price action
- **Speed Asymmetry:** Downward moves are faster/stronger than upward moves

### Does NOT Do
- ❌ Predict price direction (up/down)
- ❌ Generate trading signals
- ❌ Execute trades
- ❌ Backtest for PnL
- ❌ Provide actionable trade recommendations

### Provides
- 📊 Stress score (0.0 → 1.0)
- 🏷️ Labeled state (LOW / RISING / HIGH)
- 📈 Orthogonal context streams (open interest, funding rates)
- 📅 Timestamp-aligned snapshot for situational awareness

---

## Architecture

```
stop-hunt-detector/
├── README.md
├── LICENSE
├── requirements.txt
├── attic/                      # Archived experimental scaffolding
│   ├── detector/               # Retired: slow_context.py, stress_score.py
│   ├── validation/             # Retired: backtest, correlation, reports
│   └── output/                 # Archived validation outputs
├── data/
│   ├── fetch_coinbase.py       # OHLCV data (primary data source)
│   ├── fetch_open_interest.py   # Open interest context (NEW)
│   ├── fetch_funding_rates.py  # Funding rates context (NEW)
│   └── historical/            # Cached OHLCV data
├── signals/
│   ├── volatility.py           # Volatility compression signal
│   ├── liquidity.py            # Liquidity fragility signal
│   ├── continuation.py          # Continuation failure signal
│   └── speed.py               # Speed asymmetry signal
├── detector/
│   ├── rolling_stress_score.py # Fast relative stress detector (CORE)
│   └── state_label.py         # Labeled state output
├── context/
│   └── snapshot.py            # Timestamp-aligned context snapshot (NEW)
└── output/
    ├── final_adjudication/     # Real data validation results
    └── context_snapshot.csv    # Aligned context output
```

---

## Core Engine (What Was Preserved)

### Fast Relative Stress Detector
- **File:** `detector/rolling_stress_score.py`
- **Logic:** 7-day rolling z-score normalization
- **Output:** Continuous stress score (0.0 → 1.0)
- **Validated:** 100% stress elevation on Coinbase BTC-USD (2023-2026)

### State Classification
- **File:** `detector/state_label.py`
- **States:** LOW (0.0-0.5), RISING (0.5-0.7), HIGH (0.7-1.0)
- **Purpose:** Human-readable labels for situational awareness

### Signal Components
- **Signals:** volatility, liquidity, continuation, speed
- **Weights:** Empirically tuned (35/35/15/15)
- **Normalization:** All signals normalized 0.0-1.0

---

## What Was Retired

### Slow Context Layer
- **Reason:** Events do NOT distribute across COLD/NEUTRAL/HOT regimes in real data
- **Conclusion:** Slow context from OHLCV alone adds no value
- **Location:** `attic/detector/slow_context.py`

### Dual-Scale Interpretation
- **Reason:** Fast layer provides all necessary information
- **Conclusion:** Dual-scale interpretation adds complexity without value
- **Location:** `attic/scripts/run_dual_scale_validation.py`

### Mock Data & Validation Harnesses
- **Reason:** Real data validation complete, scaffolding no longer needed
- **Location:** `attic/` (entire directory structure preserved)

---

## New: Orthogonal Context Streams

### Open Interest
- **File:** `data/fetch_open_interest.py`
- **Source:** Binance Futures public API
- **Purpose:** Track derivatives positioning
- **Usage:** Context only — NOT used in stress scoring

### Funding Rates
- **File:** `data/fetch_funding_rates.py`
- **Source:** Binance Futures public API
- **Purpose:** Track perpetual futures funding
- **Usage:** Context only — NOT used in stress scoring

### Context Snapshot
- **File:** `context/snapshot.py`
- **Purpose:** Align all streams on common timestamp
- **Output:** `output/context_snapshot.csv`
- **Note:** This is a data product, NOT a predictor

---

## Usage

### 1. Fetch OHLCV Data

```python
from data.fetch_coinbase import CoinbaseFetcher

fetcher = CoinbaseFetcher()
df = fetcher.fetch_historical('BTCUSDT', '1h', days=365)
```

### 2. Compute Stress Scores

```python
from detector.rolling_stress_score import RollingStressCalculator

calculator = RollingStressCalculator(lookback_hours=168)
df['fast_stress'] = calculator.calculate(df)
```

### 3. Add Context Streams (Optional)

```python
from data.fetch_open_interest import OpenInterestFetcher
from data.fetch_funding_rates import FundingRatesFetcher

# Fetch context
oi_fetcher = OpenInterestFetcher()
oi_df = oi_fetcher.fetch_historical('BTCUSDT', '1h', hours=720)

fr_fetcher = FundingRatesFetcher()
fr_df = fr_fetcher.fetch_historical('BTCUSDT', hours=720)

# Align context
from context.snapshot import ContextSnapshot

snapshot = ContextSnapshot()
snapshot.add_fast_stress(df[['timestamp', 'fast_stress']])
snapshot.add_open_interest(oi_df)
snapshot.add_funding_rates(fr_df)
aligned_df = snapshot.align()
```

---

## Stress Score & States

| Stress Score | State | Interpretation |
|--------------|-------|----------------|
| 0.0 - 0.5 | LOW | Calm market, no stress indicators |
| 0.5 - 0.7 | RISING | Moderate stress elevation |
| 0.7 - 1.0 | HIGH | High probability of instability |

---

## Validation Results (Coinbase BTC-USD, 2023-2026)

| Metric | Result |
|--------|--------|
| Fast Stress Elevation | 100% (3/3 events) |
| Slow Context Separation | ❌ None (all events in COLD) |
| False Positive Rate | 0.0% |
| High-Stress Ratio | 0.04 (anticipatory, not saturated) |

**Verdict:** PASS (fast stress anticipatory) but slow context hypothesis **FAILED**.

---

## Designed For

- ✅ Situational awareness
- ✅ Risk management (deciding when to pay attention)
- ✅ Market structure analysis
- ✅ Research on stress dynamics

**NOT designed for:**
- ❌ Predictive trading
- ❌ Price direction
- ❌ Portfolio optimization
- ❌ Profit maximization

---

## Data Sources

- **Primary:** Coinbase Exchange (OHLCV)
- **Context:** Binance Futures (open interest, funding rates)
- **Product:** BTC-USD
- **Timeframe:** 1h candles

---

## License

MIT License — See [LICENSE](LICENSE)

---

## Build History

### Phase 1: Experimental Scaffolding (COMPLETED)
- Implemented fast stress detector
- Tested slow context hypothesis (❌ FAILED)
- Validated on real Coinbase data
- Archived experimental code to `attic/`

### Phase 2: Core Engine Stabilization (CURRENT)
- Preserved fast stress detector
- Removed slow context and dual-scale logic
- Added orthogonal context streams
- Created timestamp-aligned snapshot

### Phase 3: Future Work (OPTIONAL)
- [ ] Add on-chain metrics (selective, stateful only)
- [ ] Multi-asset expansion (ETH, SOL, etc.)
- [ ] Real-time streaming integration
- [ ] Web dashboard for monitoring
