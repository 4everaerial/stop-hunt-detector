"""
Empirical Signal Weight Tuning

Find optimal signal weights by testing different combinations
against simulated stress events.
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
import json

from data.mock_generator import generate_stress_event_data
from detector.stress_score import StressCalculator
from detector.state_label import StateLabeler


class WeightTuner:
    """
    Tunes signal weights empirically using simulated data.

    Tests weight combinations and evaluates:
    - Stress elevation before events
    - Signal-to-noise ratio
    - State transition quality
    """

    def __init__(
        self,
        event_day: int = 30,
        days: int = 60,
        base_price: float = 60000
    ):
        """
        Initialize weight tuner.

        Args:
            event_day: Day when simulated event occurs
            days: Total days of data
            base_price: Starting price
        """
        self.event_day = event_day
        self.days = days
        self.base_price = base_price

        # Generate base data once
        print("Generating simulated data with stress event...")
        self.df = generate_stress_event_data(event_day=event_day, days=days, base_price=base_price)

        # Event time
        self.event_time = pd.to_datetime(self.df['timestamp'].iloc[0]) + pd.Timedelta(days=event_day)

    def evaluate_weights(
        self,
        weights: Dict[str, float]
    ) -> Dict:
        """
        Evaluate a specific weight combination.

        Args:
            weights: Signal weights dict

        Returns:
            Evaluation metrics
        """
        # Calculate stress scores with these weights
        calculator = StressCalculator(weights=weights)

        # We need to re-calculate all signals
        df = self.df.copy()
        df['signal_volatility'] = calculator.volatility.calculate(df)
        df['signal_liquidity'] = calculator.liquidity.calculate(df)
        df['signal_continuation'] = calculator.continuation.calculate(df)
        df['signal_speed'] = calculator.speed.calculate(df)

        # Calculate composite
        df['stress_score'] = (
            df['signal_volatility'] * weights['volatility'] +
            df['signal_liquidity'] * weights['liquidity'] +
            df['signal_continuation'] * weights['continuation'] +
            df['signal_speed'] * weights['speed']
        )

        # Label states
        labeler = StateLabeler()
        df['state'] = labeler.label_series(df['stress_score'])

        # Calculate metrics
        metrics = self._calculate_metrics(df, weights)

        return metrics

    def _calculate_metrics(self, df: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """
        Calculate evaluation metrics.

        Args:
            df: DataFrame with stress scores
            weights: Signal weights

        Returns:
            Metrics dict
        """
        # Event time index
        event_idx = df[df['timestamp'] == self.event_time].index
        if len(event_idx) == 0:
            event_idx = df[df['timestamp'].dt.date == self.event_time.date()].index
            if len(event_idx) == 0:
                event_idx = [len(df) // 2]  # Fallback to middle
        event_idx = event_idx[0]

        # Pre-event window (4 hours before)
        pre_start = max(0, event_idx - 4)
        pre_event_stress = df.loc[pre_start:event_idx, 'stress_score'].mean()
        pre_event_max = df.loc[pre_start:event_idx, 'stress_score'].max()

        # Event stress
        event_stress = df.loc[event_idx, 'stress_score']

        # Baseline (48 hours before event)
        baseline_start = max(0, event_idx - 48)
        baseline_end = max(0, event_idx - 4)
        baseline_stress = df.loc[baseline_start:baseline_end, 'stress_score'].mean()

        # Stress elevation
        stress_elevation = pre_event_max - baseline_stress
        elevation_ratio = pre_event_max / baseline_stress if baseline_stress > 0 else 1.0

        # Signal-to-noise ratio
        signal_mask = df['stress_score'] >= 0.7
        signal_mean = df.loc[signal_mask, 'stress_score'].mean() if signal_mask.any() else 0
        noise_mean = df.loc[~signal_mask, 'stress_score'].mean()
        signal_to_noise = (signal_mean - noise_mean) / noise_mean if noise_mean > 0 else 0

        # High-stress period detection
        high_stress_count = signal_mask.sum()
        high_stress_pct = (high_stress_count / len(df)) * 100

        # State distribution
        state_dist = df['state'].value_counts(normalize=True).to_dict()

        # Composite score (higher is better)
        # Priority: stress elevation > 0.3, S/N > 1.0, reasonable high-stress %
        composite_score = 0

        # Stress elevation (target: >0.3)
        if stress_elevation > 0.3:
            composite_score += 0.4
            composite_score += min(stress_elevation - 0.3, 0.3) / 0.3 * 0.1  # Up to 0.1 bonus

        # S/N ratio (target: >1.0)
        if signal_to_noise > 1.0:
            composite_score += 0.3
            composite_score += min(signal_to_noise - 1.0, 1.0) / 1.0 * 0.1  # Up to 0.1 bonus

        # High-stress % (target: 5-15%)
        if 5 <= high_stress_pct <= 15:
            composite_score += 0.2
        elif high_stress_pct > 0:
            composite_score += 0.1 * (1 - abs(high_stress_pct - 10) / 10)  # Distance penalty

        # Reward state diversity
        if len(state_dist) >= 3:
            composite_score += 0.1

        # Reward balanced weights (not all weight on one signal)
        weight_std = np.std(list(weights.values()))
        if weight_std < 0.15:  # Reasonable balance
            composite_score += 0.1

        return {
            'weights': weights,
            'pre_event_stress': pre_event_stress,
            'pre_event_max': pre_event_max,
            'event_stress': event_stress,
            'baseline_stress': baseline_stress,
            'stress_elevation': stress_elevation,
            'elevation_ratio': elevation_ratio,
            'signal_to_noise': signal_to_noise,
            'high_stress_pct': high_stress_pct,
            'state_distribution': state_dist,
            'composite_score': composite_score
        }

    def grid_search(self, step: float = 0.1) -> List[Dict]:
        """
        Perform grid search over weight combinations.

        Args:
            step: Weight step size

        Returns:
            List of evaluation results sorted by composite score
        """
        results = []

        # Generate weight combinations (4 signals)
        # This is a simplified grid search - in practice you'd use optimization
        weight_values = np.arange(0.1, 0.8 + step, step)

        # Sample combinations (not exhaustive - too many)
        # Use a smarter approach: iterate through a manageable subset
        count = 0
        for vol in weight_values:
            for liq in weight_values:
                for cont in weight_values:
                    speed = 1.0 - vol - liq - cont
                    if speed <= 0:
                        continue

                    weights = {
                        'volatility': vol,
                        'liquidity': liq,
                        'continuation': cont,
                        'speed': speed
                    }

                    # Normalize to ensure sum = 1.0
                    total = sum(weights.values())
                    weights = {k: v/total for k, v in weights.items()}

                    # Evaluate
                    metrics = self.evaluate_weights(weights)
                    results.append(metrics)

                    count += 1
                    if count % 50 == 0:
                        print(f"  Tested {count} weight combinations...")

        # Sort by composite score
        results = sorted(results, key=lambda x: x['composite_score'], reverse=True)

        return results

    def test_preset_weights(self) -> List[Dict]:
        """
        Test a set of preset weight combinations.

        Returns:
            List of evaluation results
        """
        presets = [
            # Equal weights
            {'volatility': 0.25, 'liquidity': 0.25, 'continuation': 0.25, 'speed': 0.25},

            # Volatility-focused
            {'volatility': 0.40, 'liquidity': 0.20, 'continuation': 0.20, 'speed': 0.20},

            # Liquidity-focused
            {'volatility': 0.20, 'liquidity': 0.40, 'continuation': 0.20, 'speed': 0.20},

            # Balanced (current default)
            {'volatility': 0.30, 'liquidity': 0.30, 'continuation': 0.20, 'speed': 0.20},

            # Stress-focused (volatility + liquidity)
            {'volatility': 0.35, 'liquidity': 0.35, 'continuation': 0.15, 'speed': 0.15},

            # Momentum-focused (continuation + speed)
            {'volatility': 0.20, 'liquidity': 0.20, 'continuation': 0.30, 'speed': 0.30},
        ]

        results = []
        for i, weights in enumerate(presets):
            print(f"\nTesting preset {i+1}: {weights}")
            metrics = self.evaluate_weights(weights)
            results.append(metrics)

            print(f"  Composite score: {metrics['composite_score']:.3f}")
            print(f"  Stress elevation: {metrics['stress_elevation']:.3f}")
            print(f"  S/N ratio: {metrics['signal_to_noise']:.3f}")
            print(f"  High-stress %: {metrics['high_stress_pct']:.1f}%")

        return sorted(results, key=lambda x: x['composite_score'], reverse=True)

    def save_results(self, results: List[Dict], output_file: str):
        """
        Save tuning results to JSON.

        Args:
            results: Evaluation results
            output_file: Output file path
        """
        # Convert to serializable format
        serializable = []
        for result in results:
            serializable.append({
                'weights': result['weights'],
                'composite_score': result['composite_score'],
                'stress_elevation': result['stress_elevation'],
                'elevation_ratio': result['elevation_ratio'],
                'signal_to_noise': result['signal_to_noise'],
                'high_stress_pct': result['high_stress_pct'],
                'pre_event_max': result['pre_event_max'],
                'baseline_stress': result['baseline_stress']
            })

        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

    def print_top_results(self, results: List[Dict], top_n: int = 5):
        """
        Print top results.

        Args:
            results: Evaluation results
            top_n: Number of top results to print
        """
        print(f"\n{'='*80}")
        print(f"TOP {top_n} WEIGHT COMBINATIONS")
        print(f"{'='*80}\n")

        for i, result in enumerate(results[:top_n]):
            print(f"#{i+1} - Composite Score: {result['composite_score']:.3f}")
            print(f"  Weights: {result['weights']}")
            print(f"  Stress Elevation: {result['stress_elevation']:.3f} (ratio: {result['elevation_ratio']:.2f}x)")
            print(f"  S/N Ratio: {result['signal_to_noise']:.3f}")
            print(f"  High-Stress %: {result['high_stress_pct']:.1f}%")
            print(f"  Baseline: {result['baseline_stress']:.3f} → Pre-Event Max: {result['pre_event_max']:.3f}")
            print()


if __name__ == "__main__":
    # Initialize tuner
    tuner = WeightTuner(event_day=30, days=60)

    # Test preset weights
    print("\n" + "="*80)
    print("TESTING PRESET WEIGHT COMBINATIONS")
    print("="*80)
    results = tuner.test_preset_weights()

    # Print top results
    tuner.print_top_results(results, top_n=5)

    # Save results
    tuner.save_results(results, '/home/ross/.openclaw/workspace/stop-hunt-detector/output/weight_tuning_results.json')

    print("\n✓ Weight tuning complete!")
