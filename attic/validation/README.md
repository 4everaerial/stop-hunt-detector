# Validation

This directory contains validation and backtesting tools for the stop-hunt detector.

## Structure

- `backtest.py` - Backtesting framework for historical validation
- `correlation.py` - Correlation metrics between stress scores and liquidation events
- `report.py` - Findings report generation

## Validation Metrics

- **Stress elevation:** % of events with stress > 0.7 within 1-4 hours before liquidation
- **Correlation coefficient:** Pearson correlation between stress scores and event occurrence
- **Signal-to-noise:** Stress scores during calm markets vs pre-event

## Success Criteria

- **Stress elevation:** ≥70% of events show stress > 0.7 within 1-4 hours before
- **Correlation:** ≥0.3 between stress scores and liquidation events
- **Signal-to-noise:** Low baseline stress during calm markets (<0.3 average)
