"""
Context Stream Utility Analysis - Rigorous Falsification Framework

Tests each orthogonal context stream (OI, funding, on-chain)
against falsifiable criteria for usefulness.

NO SCORING. NO MODIFICATION OF CORE ENGINE.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')


class ContextAnalyzer:
    """Analyze context streams against falsifiable tests."""

    def __init__(self, stress_df, inflections_df):
        self.stress_df = stress_df.copy()
        self.inflections_df = inflections_df.copy()
        self.stress_df['timestamp'] = pd.to_datetime(self.stress_df['timestamp'], utc=True)
        self.inflections_df['timestamp'] = pd.to_datetime(self.inflections_df['timestamp'], utc=True)

        # Identify false positives and ambiguous regions
        self._identify_problem_regions()

    def _identify_problem_regions(self):
        """Find false positives and ambiguous stress periods."""

        # False positives: stress > 0.7 during extended calm
        self.stress_df['rolling_mean_stress'] = self.stress_df['fast_stress'].rolling(window=24, min_periods=24).mean()
        self.stress_df['is_calm'] = self.stress_df['rolling_mean_stress'] < 0.4
        self.stress_df['is_false_positive'] = (self.stress_df['is_calm']) & (self.stress_df['fast_stress'] > 0.7)

        # Ambiguous: stress in 0.5-0.7 sustained > 24h
        self.stress_df['is_mid'] = (self.stress_df['fast_stress'] >= 0.5) & (self.stress_df['fast_stress'] < 0.7)
        self.stress_df['mid_sustained'] = self.stress_df['is_mid'].rolling(window=24).sum() >= 20

    def test_1_lead_lag(self, context_df, context_col):
        """
        TEST 1: Lead-Lag Analysis

        Hypothesis: Context shows significant change BEFORE stress inflection.

        Falsification:
        - If no significant lead exists, context provides NO advantage.
        - Significance: KS test p < 0.05 OR mean shift > 2× std

        Returns:
            dict with lead_times, p_values, falsification result
        """
        results = {
            'test': 'Lead-Lag',
            'context_col': context_col,
            'passes': False,
            'evidence': {},
            'falsification': 'Context shows no significant lead before stress inflections'
        }

        if context_df is None or context_df.empty:
            results['falsification'] = 'No data available'
            return results

        # Merge context with stress
        merged = pd.merge_asof(
            self.stress_df[['timestamp', 'fast_stress']],
            context_df[['timestamp', context_col]],
            on='timestamp',
            direction='nearest'
        )

        merged = merged.dropna(subset=[context_col])

        if len(merged) < 100:
            results['falsification'] = 'Insufficient data after merge'
            return results

        # Calculate context change points (z-score spike > 2.5)
        context_z = (merged[context_col] - merged[context_col].rolling(168).mean()) / merged[context_col].rolling(168).std()
        merged['context_spike'] = np.abs(context_z) > 2.5

        # For each inflection, check for context lead
        lead_times = []
        for _, inflection in self.inflections_df.iterrows():
            inflection_time = inflection['timestamp']

            # Find context spikes in [-12h, +2h] window around inflection
            window_start = inflection_time - pd.Timedelta(hours=12)
            window_end = inflection_time + pd.Timedelta(hours=2)

            window_data = merged[(merged['timestamp'] >= window_start) & (merged['timestamp'] <= window_end)]

            if not window_data.empty:
                # Find first context spike before inflection
                spikes = window_data[window_data['context_spike'] & (window_data['timestamp'] < inflection_time)]

                if not spikes.empty:
                    lead = (inflection_time - spikes['timestamp'].iloc[-1]).total_seconds() / 3600
                    lead_times.append(lead)

        # Calculate lead time statistics
        if lead_times:
            results['evidence']['avg_lead_hours'] = np.mean(lead_times)
            results['evidence']['median_lead_hours'] = np.median(lead_times)
            results['evidence']['coverage_pct'] = len(lead_times) / len(self.inflections_df) * 100

            # Test significance: is lead distribution significantly positive?
            if len(lead_times) >= 5:
                # One-sample t-test against null hypothesis: mean lead = 0
                t_stat, p_val = stats.ttest_1samp(lead_times, 0)

                results['evidence']['t_statistic'] = t_stat
                results['evidence']['p_value'] = p_val

                # Falsification: if p >= 0.05 OR mean lead <= 0, no significant lead
                if p_val < 0.05 and np.mean(lead_times) > 0:
                    results['passes'] = True
                    results['falsification'] = None
                else:
                    results['passes'] = False
                    results['falsification'] = f'No significant lead: p={p_val:.4f}, mean lead={np.mean(lead_times):.2f}h'
        else:
            results['falsification'] = 'No context spikes detected near inflections'

        return results

    def test_2_residuals(self, context_df, context_col):
        """
        TEST 2: Residual Explanation

        Hypothesis: Context shows abnormal state where core engine struggles (false positives, ambiguous).

        Falsification:
        - If context in these regions is within normal 95% CI, adds no explanatory value.
        - Abnormal defined as outside 95% CI of baseline

        Returns:
            dict with abnormality tests, falsification result
        """
        results = {
            'test': 'Residual Explanation',
            'context_col': context_col,
            'passes': False,
            'evidence': {},
            'falsification': 'Context is within normal range during problematic regions'
        }

        if context_df is None or context_df.empty:
            results['falsification'] = 'No data available'
            return results

        # Merge context with stress
        merged = pd.merge_asof(
            self.stress_df,
            context_df[['timestamp', context_col]],
            on='timestamp',
            direction='nearest'
        )

        merged = merged.dropna(subset=[context_col])

        if len(merged) < 100:
            results['falsification'] = 'Insufficient data after merge'
            return results

        # Calculate baseline distribution (exclude problem regions)
        baseline = merged[~merged['is_false_positive'] & ~merged['mid_sustained']]
        baseline_mean = baseline[context_col].mean()
        baseline_std = baseline[context_col].std()
        ci_95_lower = baseline_mean - 1.96 * baseline_std
        ci_95_upper = baseline_mean + 1.96 * baseline_std

        # Test false positive regions
        fp_regions = merged[merged['is_false_positive']]

        if len(fp_regions) > 0:
            fp_outside_ci = (fp_regions[context_col] < ci_95_lower) | (fp_regions[context_col] > ci_95_upper)
            fp_abnormality_rate = fp_outside_ci.sum() / len(fp_regions) * 100

            results['evidence']['fp_abnormality_rate'] = fp_abnormality_rate
            results['evidence']['fp_baseline_mean'] = baseline_mean
            results['evidence']['fp_mean'] = fp_regions[context_col].mean()

        # Test ambiguous regions
        ambiguous_regions = merged[merged['mid_sustained']]

        if len(ambiguous_regions) > 0:
            amb_outside_ci = (ambiguous_regions[context_col] < ci_95_lower) | (ambiguous_regions[context_col] > ci_95_upper)
            amb_abnormality_rate = amb_outside_ci.sum() / len(ambiguous_regions) * 100

            results['evidence']['ambiguous_abnormality_rate'] = amb_abnormality_rate
            results['evidence']['ambiguous_baseline_mean'] = baseline_mean
            results['evidence']['ambiguous_mean'] = ambiguous_regions[context_col].mean()

        # Falsification: if abnormality rate < 20%, context provides no explanatory value
        total_abnormality = 0
        total_samples = 0

        if 'fp_abnormality_rate' in results['evidence']:
            total_abnormality += fp_abnormality_rate * len(fp_regions)
            total_samples += len(fp_regions)

        if 'ambiguous_abnormality_rate' in results['evidence']:
            total_abnormality += amb_abnormality_rate * len(ambiguous_regions)
            total_samples += len(ambiguous_regions)

        if total_samples > 0:
            overall_abnormality = total_abnormality / total_samples

            if overall_abnormality >= 20:
                results['passes'] = True
                results['falsification'] = None
            else:
                results['passes'] = False
                results['falsification'] = f'Low abnormality: {overall_abnormality:.1f}% < 20% threshold'
        else:
            results['falsification'] = 'No problem regions detected'

        return results

    def test_3_regime_discrimination(self, context_df, context_col):
        """
        TEST 3: Regime Discrimination

        Hypothesis: Context separates regimes where price/ohlcv appear similar.

        Falsification:
        - If Jensen-Shannon divergence between regimes <= 0.1, adds no differentiation.

        Returns:
            dict with JS divergence, falsification result
        """
        results = {
            'test': 'Regime Discrimination',
            'context_col': context_col,
            'passes': False,
            'evidence': {},
            'falsification': 'Context does not separate regimes (JS divergence <= 0.1)'
        }

        if context_df is None or context_df.empty:
            results['falsification'] = 'No data available'
            return results

        # Merge context with stress
        merged = pd.merge_asof(
            self.stress_df,
            context_df[['timestamp', context_col]],
            on='timestamp',
            direction='nearest'
        )

        merged = merged.dropna(subset=[context_col])

        if len(merged) < 500:
            results['falsification'] = 'Insufficient data for regime analysis'
            return results

        # Cluster by stress regime: LOW (<0.5), RISING (0.5-0.7), HIGH (>0.7)
        merged['stress_regime'] = pd.cut(
            merged['fast_stress'],
            bins=[0, 0.5, 0.7, 1.0],
            labels=['LOW', 'RISING', 'HIGH'],
            include_lowest=True
        )

        # Calculate distribution of context in each regime
        regime_dists = {}

        for regime in ['LOW', 'RISING', 'HIGH']:
            regime_data = merged[merged['stress_regime'] == regime][context_col].dropna()

            if len(regime_data) > 0:
                # Create histogram bins
                hist, bins = np.histogram(regime_data, bins=50, density=True)
                regime_dists[regime] = (hist, bins)

        # Calculate Jensen-Shannon divergence between regimes
        js_divergences = {}

        # LOW vs HIGH (most meaningful separation)
        if 'LOW' in regime_dists and 'HIGH' in regime_dists:
            p_low, bins = regime_dists['LOW']
            p_high, _ = regime_dists['HIGH']

            # Normalize to probability distributions
            p_low = p_low + 1e-10
            p_high = p_high + 1e-10
            p_low = p_low / p_low.sum()
            p_high = p_high / p_high.sum()

            # Add small bins to match
            m = (p_low + p_high) / 2

            # KL divergence
            kl_low_high = np.sum(p_low * np.log(p_low / m))
            kl_high_low = np.sum(p_high * np.log(p_high / m))

            js_low_high = (kl_low_high + kl_high_low) / 2
            js_divergences['LOW_vs_HIGH'] = js_low_high

        results['evidence']['js_divergences'] = js_divergences

        # Falsification: if JS divergence <= 0.1, no meaningful separation
        max_js = max(js_divergences.values()) if js_divergences else 0

        if max_js > 0.1:
            results['passes'] = True
            results['falsification'] = None
        else:
            results['passes'] = False
            results['falsification'] = f'No regime separation: max JS divergence = {max_js:.4f} <= 0.1'

        return results

    def generate_report(self, oi_df, funding_df, onchain_df):
        """
        Generate falsification report for all context streams.
        """
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'streams': {}
        }

        # Test Open Interest
        if oi_df is not None and not oi_df.empty:
            report['streams']['open_interest'] = {
                'test_1_lead_lag': self.test_1_lead_lag(oi_df, 'open_interest'),
                'test_2_residuals': self.test_2_residuals(oi_df, 'open_interest'),
                'test_3_regime': self.test_3_regime_discrimination(oi_df, 'open_interest'),
            }

            # Overall OI decision
            oi_passes = sum([
                report['streams']['open_interest']['test_1_lead_lag']['passes'],
                report['streams']['open_interest']['test_2_residuals']['passes'],
                report['streams']['open_interest']['test_3_regime']['passes']
            ])

            if oi_passes >= 2:
                report['streams']['open_interest']['recommendation'] = 'KEEP - Provides measurable value'
            elif oi_passes == 1:
                report['streams']['open_interest']['recommendation'] = 'KEEP - Weak evidence, consider retention'
            else:
                report['streams']['open_interest']['recommendation'] = 'ARCHIVE - No measurable value'

        # Test Funding Rates
        if funding_df is not None and not funding_df.empty:
            report['streams']['funding_rates'] = {
                'test_1_lead_lag': self.test_1_lead_lag(funding_df, 'funding_rate'),
                'test_2_residuals': self.test_2_residuals(funding_df, 'funding_rate'),
                'test_3_regime': self.test_3_regime_discrimination(funding_df, 'funding_rate'),
            }

            funding_passes = sum([
                report['streams']['funding_rates']['test_1_lead_lag']['passes'],
                report['streams']['funding_rates']['test_2_residuals']['passes'],
                report['streams']['funding_rates']['test_3_regime']['passes']
            ])

            if funding_passes >= 2:
                report['streams']['funding_rates']['recommendation'] = 'KEEP - Provides measurable value'
            elif funding_passes == 1:
                report['streams']['funding_rates']['recommendation'] = 'KEEP - Weak evidence, consider retention'
            else:
                report['streams']['funding_rates']['recommendation'] = 'ARCHIVE - No measurable value'

        # Test On-Chain
        if onchain_df is not None and not onchain_df.empty:
            # Test first metric
            onchain_col = None
            for col in onchain_df.columns:
                if col != 'timestamp':
                    onchain_col = col
                    break

            if onchain_col:
                report['streams']['onchain'] = {
                    'test_1_lead_lag': self.test_1_lead_lag(onchain_df, onchain_col),
                    'test_2_residuals': self.test_2_residuals(onchain_df, onchain_col),
                    'test_3_regime': self.test_3_regime_discrimination(onchain_df, onchain_col),
                }

                onchain_passes = sum([
                    report['streams']['onchain']['test_1_lead_lag']['passes'],
                    report['streams']['onchain']['test_2_residuals']['passes'],
                    report['streams']['onchain']['test_3_regime']['passes']
                ])

                if onchain_passes >= 2:
                    report['streams']['onchain']['recommendation'] = 'ARCHIVE - Context streams do not modify core engine'
                elif onchain_passes == 1:
                    report['streams']['onchain']['recommendation'] = 'ARCHIVE - Weak evidence, context streams do not modify core engine'
                else:
                    report['streams']['onchain']['recommendation'] = 'ARCHIVE - No measurable value'
            else:
                report['streams']['onchain'] = {
                    'recommendation': 'ARCHIVE - No usable metrics found'
                }

        return report


def main():
    """Run full analysis pipeline."""
    print("="*100)
    print("CONTEXT STREAM UTILITY ANALYSIS - FALSIFICATION FRAMEWORK")
    print("="*100)
    print()

    # Load stress data and inflections
    stress_df = pd.read_csv('output/final_adjudication/scores_timeseries.csv')
    stress_df['timestamp'] = pd.to_datetime(stress_df['timestamp'], utc=True)

    inflections_df = pd.read_csv('output/context_analysis/inflections.csv')
    inflections_df['timestamp'] = pd.to_datetime(inflections_df['timestamp'], utc=True)

    print(f"Loaded stress data: {len(stress_df)} records")
    print(f"Loaded inflections: {len(inflections_df)} points")
    print()

    # Try to load context data
    oi_df = None
    funding_df = None
    onchain_df = None

    # Try OI
    try:
        oi_df = pd.read_csv('data/historical/BTCUSDT_1h_open_interest.csv')
        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], utc=True)
        print(f"✓ Loaded Open Interest: {len(oi_df)} records")
    except FileNotFoundError:
        print("⚠ Open Interest not found (need to fetch)")

    # Try funding
    try:
        funding_df = pd.read_csv('data/historical/BTCUSDT_funding_rate.csv')
        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], utc=True)
        print(f"✓ Loaded Funding Rates: {len(funding_df)} records")
    except FileNotFoundError:
        print("⚠ Funding Rates not found (need to fetch)")

    # Try on-chain
    try:
        onchain_df = pd.read_csv('data/historical/btc_1d_onchain.csv')
        onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'], utc=True)
        print(f"✓ Loaded On-Chain: {len(onchain_df)} records")
    except FileNotFoundError:
        print("⚠ On-Chain data not found (need to fetch)")

    print()

    # Run analysis
    analyzer = ContextAnalyzer(stress_df, inflections_df)
    report = analyzer.generate_report(oi_df, funding_df, onchain_df)

    # Save report
    output_path = 'output/context_analysis/falsification_report.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"✓ Saved report: {output_path}")
    print()

    # Print summary
    print("="*100)
    print("SUMMARY")
    print("="*100)

    for stream_name, stream_data in report.get('streams', {}).items():
        print(f"\n{stream_name.upper()}:")
        print(f"  Recommendation: {stream_data.get('recommendation', 'N/A')}")

        for test_name, test_result in stream_data.items():
            if test_name != 'recommendation':
                status = "✅ PASS" if test_result['passes'] else "❌ FAIL"
                print(f"  {test_name}: {status}")
                if test_result.get('falsification'):
                    print(f"    └ {test_result['falsification']}")


if __name__ == "__main__":
    main()
