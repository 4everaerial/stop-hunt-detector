# Final Adjudication Report - Coinbase BTC-USD

**Generated:** 2026-02-10T02:18:11.498168+00:00
**Verdict:** ✅ PASS

---

## Data Source
- Exchange: Coinbase
- Product: BTC-USD
- Timeframe: 1h
- Data Range: 2023-01-01 00:00:00+00:00 to 2026-02-10 00:00:00+00:00
- Total Candles: 27257

## Metrics
- Fast Stress Elevation Rate: 100.0% (3/3)
- Avg Fast Stress @ Events: nan
- Avg Slow Context @ Events: 0.000
- False Positive Rate (calm periods): 0.0%

### Slow Context Distribution @ Events
- COLD: 3 events (100.0%)
- NEUTRAL: 0 events (0.0%)
- HOT: 0 events (0.0%)

### 2D State Table (Event Counts)
- HIGH×COLD: 3 events (100.0%)

## Plots
- fast_stress_over_time: /home/ross/.openclaw/workspace/stop-hunt-detector/output/final_adjudication/fast_stress_over_time.png
- slow_context_over_time: /home/ross/.openclaw/workspace/stop-hunt-detector/output/final_adjudication/slow_context_over_time.png
- fast_vs_slow_scatter: /home/ross/.openclaw/workspace/stop-hunt-detector/output/final_adjudication/fast_vs_slow_scatter.png
- state_table_counts: /home/ross/.openclaw/workspace/stop-hunt-detector/output/final_adjudication/state_table_counts.png

## One-Paragraph Interpretation
Coinbase BTC-USD data shows no slow context separation: all events fall in a single slow regime (COLD). Fast stress remains anticipatory without saturation (high-stress ratio 0.04) and false positives during calm periods are 0.0%, so PASS is driven solely by the fast-stress criterion.
