# Stop-Hunt Detector — Market Stress Detection System

**Experiment ID:** `f3c1c4d4`  
**Status:** ✅ Day 1 Complete  
**Timebox:** 7 days build + 7 days validate

**Repository:** https://github.com/4everaerial/stop-hunt-detector

---

## Overview

A Python-based market stress detector that estimates the probability of imminent forced-liquidation/stop-hunt events via a continuous stress score (0.0→1.0) and labeled state output.

**This is NOT a trading bot.** It does not generate buy/sell signals, predict price direction, or execute trades. It measures market stress conditions that often precede forced liquidations.

---

## Core Concept

Stop hunts occur when price briefly spikes to trigger leveraged positions' stop-loss orders, then reverses. These events are characterized by:

1. **Volatility Compression:** Tight trading ranges followed by explosive expansion
2. **Liquidity Fragility:** Order book thinning, large wicks on small volume
3. **Continuation Failure:** Momentum indicators diverge from price action
4. **Speed Asymmetry:** Downward moves are faster/stronger than upward moves

The detector aggregates these signals into a composite stress score (0.0→1.0) and outputs a labeled state.

---

## Stress Score & States

| Stress Score | State | Interpretation |
|--------------|-------|----------------|
| 0.0 - 0.3 | `NORMAL` | Calm market, no stress indicators |
| 0.3 - 0.6 | `STRESSED` | Moderate compression or fragility |
| 0.6 - 0.8 | `FRAGILE` | Multiple stress conditions present |
| 0.8 - 1.0 | `IMMINENT_CLEARING` | High probability of forced liquidation event |

---

## Project Structure

```
stop-hunt-detector/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── fetch_binance.py      # OHLCV data fetching
│   └── tag_events.py         # Manual liquidation event tagging
├── signals/
│   ├── __init__.py
│   ├── volatility.py         # Volatility compression signal
│   ├── liquidity.py          # Liquidity fragility signal
│   ├── continuation.py       # Continuation failure signal
│   └── speed.py              # Speed asymmetry signal
├── detector/
│   ├── __init__.py
│   ├── stress_score.py       # Composite stress score calculation
│   └── state_label.py        # Labeled state output
├── validation/
│   ├── backtest.py           # Historical validation
│   ├── correlation.py        # Correlation metrics
│   └── report.py             # Findings report
└── events/
    └── tagged_events.json    # Liquidation events
```

---

## Usage

### 1. Fetch Data

```python
from data.fetch_binance import BinanceFetcher

fetcher = BinanceFetcher()
fetcher.fetch_historical('BTCUSDT', '1h', days=365)
```

### 2. Calculate Stress Score

```python
from detector.stress_score import StressCalculator

calculator = StressCalculator()
stress = calculator.calculate_score(df)  # Returns 0.0-1.0
state = calculator.get_state(stress)      # Returns labeled state
```

### 3. Run Full Pipeline

```python
python validation/backtest.py
```

---

## Detailed Methodology

### Signal Calculations

**1. Volatility Compression (35% weight)**
- Bollinger Band squeeze: Width = (Upper - Lower) / SMA
- ATR decay: True Range normalized by rolling window
- Range contraction: (High - Low) / Close
- Inverse normalized → tight ranges → high compression

**2. Liquidity Fragility (35% weight)**
- Wick-to-body ratio: (Upper wick + Lower wick) / Body
- Volume-to-range ratio: Volume / (High - Low)
- Rejection magnitude: (Upper wick + Lower wick) / Total range
- High ratios → thin order books → fragility

**3. Continuation Failure (15% weight)**
- RSI divergence: Price makes high, RSI doesn't
- MACD divergence: Price vs MACD histogram mismatch
- Momentum deceleration: Price change vs ROC ratio
- Divergence → weak momentum → continuation failure

**4. Speed Asymmetry (15% weight)**
- Downward velocity: Sum of negative candles / window
- Upward velocity: Sum of positive candles / window
- Magnitude asymmetry: Down move size / total move size
- Candle count bias: Down candles / total candles
- Downward bias → panic selling/liquidations

### Stress Score Computation

```
Stress Score = 0.35 × Volatility +
              0.35 × Liquidity +
              0.15 × Continuation +
              0.15 × Speed
```

All signals normalized to 0.0 → 1.0 before weighting.

### State Thresholds

| State | Stress Range | Interpretation |
|-------|-------------|---------------|
| NORMAL | 0.0 - 0.3 | Calm market, no stress indicators |
| STRESSED | 0.3 - 0.6 | Moderate compression or fragility |
| FRAGILE | 0.6 - 0.8 | Multiple stress conditions present |
| IMMINENT_CLEARING | 0.8 - 1.0 | High probability of forced liquidation |

---

## Scripts & Tools

| Script | Purpose |
|--------|---------|
| `scripts/visualize_stress.py` | Generate stress score plots, distribution charts, heatmaps |
| `scripts/tune_weights.py` | Empirical weight tuning on simulated data |
| `scripts/full_pipeline_demo.py` | Demonstrate complete detector pipeline from data to report |
| `data/mock_generator.py` | Generate basic synthetic stress events for testing |
| `data/mock_generator_enhanced.py` | Generate enhanced realistic stress event data |

---

## Validation Plan

**Success Criteria:**

- **Stress Elevation:** ≥70% of events show stress > 0.7 within 1-4 hours before
- **Correlation:** ≥0.3 between stress scores and liquidation events
- **Signal-to-Noise:** Low baseline stress during calm markets (<0.3 average)

**Kill Criteria (Day 7):**

- Correlation < 0.3
- <50% of events show stress elevation
- Stress scores noisy or indiscriminately elevated

---

## Data Sources

- **Primary:** Binance Public API (OHLCV)
- **Pairs:** BTCUSDT, ETHUSDT, SOLUSDT, etc.
- **Timeframes:** 1h (primary), 15m (high-res validation)

---

## License

MIT License — See [LICENSE](LICENSE)

---

## Build Status

**Day 1 (Complete):**
- [x] Project structure
- [x] Data ingestion pipeline (fetch_binance.py + mock_generator.py for testing)
- [x] Volatility compression signal
- [x] Liquidity fragility signal
- [x] Continuation failure signal
- [x] Speed asymmetry signal
- [x] Basic stress score composition
- [x] Labeled state output
- [x] Validation framework (backtest, correlation, report)

**Day 2 (Complete):**
- [x] Empirical weight tuning (stress-focused: 35/35/15/15)
- [x] Stress score visualizations (main plot, distribution, heatmap)
- [x] Full pipeline demonstration script
- [x] Methodology documentation
- [x] Enhanced mock data generator

**Day 3 (Upcoming):**
- [ ] Historical data pipeline for real validation
- [ ] Event tagging system (10-20 historical liquidation events)
- [ ] Backtesting on multiple events
- [ ] Correlation metrics refinement
