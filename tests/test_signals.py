"""
Unit tests for market stress signals.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.volatility import VolatilitySignal
from signals.liquidity import LiquiditySignal
from signals.continuation import ContinuationSignal
from signals.speed import SpeedSignal
from detector.stress_score import StressScoreDetector
from detector.state_label import StateLabeler


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1h'),
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.1) + np.random.rand(n) * 2,
        'low': 100 + np.cumsum(np.random.randn(n) * 0.1) - np.random.rand(n) * 2,
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': 1000 + np.random.rand(n) * 100
    }

    df = pd.DataFrame(data)
    # Ensure high >= close and low <= close
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)

    return df


class TestVolatilitySignal:
    """Test volatility compression signal."""

    def test_calculate_atr(self, sample_data):
        """Test ATR calculation."""
        from signals.volatility import calculate_atr

        atr = calculate_atr(sample_data, period=14)

        assert len(atr) == len(sample_data), "ATR length mismatch"
        assert atr.notna().sum() > 0, "All ATR values are NaN"
        assert atr.min() >= 0, "ATR contains negative values"

    def test_calculate_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        from signals.volatility import calculate_bollinger_bands

        upper, middle, lower, bandwidth = calculate_bollinger_bands(sample_data, period=20)

        assert len(upper) == len(sample_data), "Upper band length mismatch"
        assert len(middle) == len(sample_data), "Middle band length mismatch"
        assert len(lower) == len(sample_data), "Lower band length mismatch"
        # Check only non-NaN values
        valid_mask = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert (upper[valid_mask] >= middle[valid_mask]).all(), "Upper band below middle"
        assert (middle[valid_mask] >= lower[valid_mask]).all(), "Middle band below lower"

    def test_volatility_signal_calculate(self, sample_data):
        """Test full volatility signal calculation."""
        signal = VolatilitySignal()
        df = signal.calculate(sample_data)

        assert 'volatility_score' in df.columns, "Missing volatility_score"
        assert df['volatility_score'].notna().any(), "All volatility scores NaN"
        assert df['volatility_score'].min() >= 0.0, "Volatility score below 0"
        assert df['volatility_score'].max() <= 1.0, "Volatility score above 1"


class TestLiquiditySignal:
    """Test liquidity fragility signal."""

    def test_wick_body_ratio(self, sample_data):
        """Test wick-to-body ratio calculation."""
        from signals.liquidity import calculate_wick_body_ratio

        ratio = calculate_wick_body_ratio(sample_data)

        assert len(ratio) == len(sample_data), "Ratio length mismatch"
        assert ratio.notna().any(), "All ratios are NaN"

    def test_liquidity_signal_calculate(self, sample_data):
        """Test full liquidity signal calculation."""
        signal = LiquiditySignal()
        df = signal.calculate(sample_data)

        assert 'liquidity_score' in df.columns, "Missing liquidity_score"
        assert df['liquidity_score'].notna().any(), "All liquidity scores NaN"
        assert df['liquidity_score'].min() >= 0.0, "Liquidity score below 0"
        assert df['liquidity_score'].max() <= 1.0, "Liquidity score above 1"


class TestContinuationSignal:
    """Test continuation failure signal."""

    def test_rsi(self, sample_data):
        """Test RSI calculation."""
        from signals.continuation import calculate_rsi

        rsi = calculate_rsi(sample_data, period=14)

        assert len(rsi) == len(sample_data), "RSI length mismatch"
        assert rsi.notna().any(), "All RSI values NaN"
        assert rsi.min() >= 0, "RSI below 0"
        assert rsi.max() <= 100, "RSI above 100"

    def test_continuation_signal_calculate(self, sample_data):
        """Test full continuation signal calculation."""
        signal = ContinuationSignal()
        df = signal.calculate(sample_data)

        assert 'continuation_score' in df.columns, "Missing continuation_score"
        assert df['continuation_score'].notna().any(), "All continuation scores NaN"
        assert df['continuation_score'].min() >= 0.0, "Continuation score below 0"
        assert df['continuation_score'].max() <= 1.0, "Continuation score above 1"


class TestSpeedSignal:
    """Test speed asymmetry signal."""

    def test_velocity_asymmetry(self, sample_data):
        """Test velocity asymmetry calculation."""
        from signals.speed import calculate_velocity_asymmetry

        asymmetry = calculate_velocity_asymmetry(sample_data, period=5)

        assert len(asymmetry) == len(sample_data), "Asymmetry length mismatch"
        assert asymmetry.notna().any(), "All asymmetries are NaN"

    def test_speed_signal_calculate(self, sample_data):
        """Test full speed signal calculation."""
        signal = SpeedSignal()
        df = signal.calculate(sample_data)

        assert 'speed_score' in df.columns, "Missing speed_score"
        assert df['speed_score'].notna().any(), "All speed scores NaN"
        assert df['speed_score'].min() >= 0.0, "Speed score below 0"
        assert df['speed_score'].max() <= 1.0, "Speed score above 1"


class TestStressScoreDetector:
    """Test composite stress score detector."""

    def test_stress_score_calculate(self, sample_data):
        """Test stress score calculation."""
        # Add all signal scores
        sample_data['volatility_score'] = 0.5
        sample_data['liquidity_score'] = 0.5
        sample_data['continuation_score'] = 0.5
        sample_data['speed_score'] = 0.5

        detector = StressScoreDetector()
        df = detector.calculate_stress_score(sample_data)

        assert 'stress_score' in df.columns, "Missing stress_score"
        assert df['stress_score'].notna().any(), "All stress scores NaN"
        # All 0.5 inputs should give 0.5 output with equal weights
        assert abs(df['stress_score'].iloc[-1] - 0.5) < 0.01, "Stress score calculation incorrect"

    def test_missing_signals(self, sample_data):
        """Test behavior with missing signal columns."""
        detector = StressScoreDetector()
        df = detector.calculate_stress_score(sample_data)

        assert 'stress_score' in df.columns, "Missing stress_score"
        # Should handle missing columns gracefully
        assert df['stress_score'].notna().any(), "Stress score all NaN with missing signals"


class TestStateLabeler:
    """Test state labeler."""

    def test_state_labeling(self, sample_data):
        """Test state labeling."""
        sample_data['stress_score'] = 0.5

        labeler = StateLabeler()
        df = labeler.label_state(sample_data)

        assert 'state' in df.columns, "Missing state column"
        assert set(df['state'].unique()).issubset({'NORMAL', 'STRESSED', 'FRAGILE', 'IMMINENT_CLEARING', 'UNKNOWN'}), \
            "Invalid state labels"

    def test_threshold_logic(self):
        """Test state threshold logic."""
        data = pd.DataFrame({
            'stress_score': [0.1, 0.4, 0.7, 0.9]
        })

        labeler = StateLabeler()
        df = labeler.label_state(data)

        assert df.loc[0, 'state'] == 'NORMAL', "Normal state threshold incorrect"
        assert df.loc[1, 'state'] == 'STRESSED', "Stressed state threshold incorrect"
        assert df.loc[2, 'state'] == 'FRAGILE', "Fragile state threshold incorrect"
        assert df.loc[3, 'state'] == 'IMMINENT_CLEARING', "Imminent clearing state threshold incorrect"
