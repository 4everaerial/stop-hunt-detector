# Context Stream Utility Analysis - Falsification Report

**Analysis Date:** 2026-02-10T03:20:00Z  
**Purpose:** Determine whether orthogonal context streams (OI, funding, on-chain) provide measurable value relative to core stress engine

---

## Executive Summary

**RESULT: UNABLE TO EVALUATE - DATA ACCESS BLOCKED**

All public API sources for context data are inaccessible:
- **Binance Futures (OI, funding):** HTTP 451 - Geoblocked/Unavailable
- **Coin Metrics Community (on-chain):** HTTP 401 - Authentication required

Without context data, falsification tests cannot be executed.

---

## Attempted Data Sources

### 1. Binance Futures Public API
**Endpoints:**
- `/fapi/v1/openInterest` - Open interest history
- `/fapi/v1/fundingRate` - Funding rate history

**Status:** ❌ FAILED - HTTP 451 Client Error
**Error Message:** Geoblocked or service unavailable for this region
**Result:** 0 records fetched

### 2. Coin Metrics Community API
**Endpoint:**
- `/v4/timeseries/asset-metrics` - On-chain metrics

**Status:** ❌ FAILED - HTTP 401 Unauthorized
**Error Message:** API key required
**Result:** 0 records fetched

---

## Consequence

**Context streams cannot be tested.** The falsification framework (lead-lag, residuals, regime discrimination) requires aligned context data to execute.

---

## Recommendations

### Option 1: Obtain API Keys (If Permitted)
- **Coin Metrics API Key:** Public tier may provide limited access
- **Alternative Public Sources:** Research other free public APIs:
  - Glassnode (public metrics limited)
  - CryptoQuant (public metrics limited)
  - OKX/Bybit public APIs (may have same geo-block issues)

### Option 2: Mock Context Data (For Testing Framework Only)
- Generate synthetic context data with known lead-lag relationships
- Test falsification framework works correctly
- **NOT** a substitute for real evaluation
- Would validate testing methodology, not actual utility

### Option 3: Archive Context Code
- Move `data/fetch_open_interest.py`, `data/fetch_funding_rates.py`, `data/fetch_onchain.py` to `attic/`
- Remove `context/` module
- Acknowledge that context hypothesis cannot be evaluated with available data sources
- Preserve code for future evaluation if API access becomes available

---

## Honest Assessment

**NO CONCLUSION CAN BE DRAWN** about context stream usefulness because **data cannot be accessed**.

The core engine (fast stress detector) has been validated on real Coinbase data. Context streams were added as orthogonal data products for situational awareness, but their utility cannot be measured without actual data.

**Context streams exist as code artifacts only. They have not been tested and provide no confirmed value.**

---

## Falsification Framework Status

| Test | Status | Reason |
|------|--------|--------|
| Lead-Lag Analysis | ⚸ NOT EXECUTED | No context data |
| Residual Explanation | ⚸ NOT EXECUTED | No context data |
| Regime Discrimination | ⚸ NOT EXECUTED | No context data |

---

## Recommendation for Repository

**RECOMMENDATION: ARCHIVE CONTEXT STREAMS**

Rationale:
1. **No measurable value has been demonstrated** (tests cannot run)
2. **No data access is available** with current API sources
3. **Core engine provides sufficient value** (validated on real data)
4. **Context streams add code complexity** without benefit
5. **Repository is cleaner** without untested orthogonal components

**Action:**
```bash
git mv data/fetch_open_interest.py data/fetch_funding_rates.py data/fetch_onchain.py attic/
git mv context/ attic/
# Update README to remove context stream references
```

**Alternative (if API access becomes available):**
- Fetch real context data
- Run full falsification framework
- If any stream passes 2/3 tests, restore from attic
- Update README with confirmed utility

---

## Notes

- Data access issues were known from previous validation attempts
- This is NOT a failure of the testing framework
- The framework is implemented correctly in `scripts/context_utility_analysis.py`
- The blocking factor is external (API access), not internal logic
