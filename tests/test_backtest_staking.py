"""Unit tests for ``wta_clay.backtest.staking``."""
from __future__ import annotations

import math

import pytest

from wta_clay.backtest.staking import (
    DEFAULT_FRACTION, DEFAULT_MAX_BET_PCT,
    fractional_kelly, kelly_fraction, kelly_stake,
)


class TestKellyFraction:
    def test_zero_at_zero_ev(self):
        # prob=0.5, odds=2.0 → b=1, edge = 1·0.5 - 0.5 = 0 → f=0
        assert math.isclose(kelly_fraction(0.5, 2.0), 0.0, abs_tol=1e-12)

    def test_positive_ev_returns_positive(self):
        # prob=0.55, odds=2.0 → b=1, edge=0.10 → f=0.10
        f = kelly_fraction(0.55, 2.0)
        assert math.isclose(f, 0.10, rel_tol=1e-9)

    def test_negative_ev_returns_zero(self):
        # prob=0.40, odds=2.0 → edge=-0.20 → no-short gives 0
        assert kelly_fraction(0.40, 2.0) == 0.0

    def test_high_odds_underdog(self):
        # prob=0.30, odds=5.0 → b=4, edge=4·0.30 - 0.70 = 0.50 → f=0.125
        f = kelly_fraction(0.30, 5.0)
        assert math.isclose(f, 0.125, rel_tol=1e-9)

    def test_low_odds_favorite(self):
        # prob=0.80, odds=1.50 → b=0.5, edge=0.5·0.80 - 0.20 = 0.20 → f=0.40
        f = kelly_fraction(0.80, 1.50)
        assert math.isclose(f, 0.40, rel_tol=1e-9)

    def test_edge_scales_linearly(self):
        f1 = kelly_fraction(0.55, 2.0)   # edge 0.10
        f2 = kelly_fraction(0.60, 2.0)   # edge 0.20
        assert math.isclose(f2, 2.0 * f1, rel_tol=1e-9)

    def test_invalid_prob(self):
        with pytest.raises(ValueError):
            kelly_fraction(1.5, 2.0)
        with pytest.raises(ValueError):
            kelly_fraction(-0.1, 2.0)

    def test_invalid_odds(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.5, 1.0)
        with pytest.raises(ValueError):
            kelly_fraction(0.5, 0.5)


class TestFractionalKelly:
    def test_quarter_kelly_default(self):
        # Use a small-edge case so the 2% cap does NOT bind:
        # prob=0.52, odds=2.0 → full Kelly = 0.04, quarter = 0.01 (< cap)
        f = fractional_kelly(0.52, 2.0)
        assert math.isclose(f, 0.01, rel_tol=1e-9)

    def test_cap_dominates_when_kelly_large(self):
        # prob=0.80, odds=2.0 → full Kelly = 0.60, quarter = 0.15.
        # Cap = 0.02 → returned 0.02
        f = fractional_kelly(0.80, 2.0)
        assert math.isclose(f, DEFAULT_MAX_BET_PCT, rel_tol=1e-9)

    def test_zero_at_zero_ev(self):
        assert fractional_kelly(0.5, 2.0) == 0.0

    def test_zero_at_negative_ev(self):
        assert fractional_kelly(0.4, 2.0) == 0.0

    def test_custom_fraction(self):
        # prob=0.55, odds=2.0, half-Kelly = 0.05 (below 0.02 cap? no, 0.05>0.02)
        # → cap binds, returns 0.02
        f = fractional_kelly(0.55, 2.0, fraction=0.5)
        assert math.isclose(f, 0.02, rel_tol=1e-9)
        # With higher cap, returns 0.05
        f2 = fractional_kelly(0.55, 2.0, fraction=0.5, max_bet_pct=0.10)
        assert math.isclose(f2, 0.05, rel_tol=1e-9)

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, fraction=0.0)
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, fraction=2.0)

    def test_invalid_max_bet(self):
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, max_bet_pct=-0.01)
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, max_bet_pct=2.0)

    def test_default_fraction_is_quarter(self):
        assert DEFAULT_FRACTION == 0.25

    def test_default_cap_is_2pct(self):
        assert DEFAULT_MAX_BET_PCT == 0.02


class TestKellyStake:
    def test_stake_scales_with_bankroll(self):
        s1, f1 = kelly_stake(10000, 0.55, 2.0)
        s2, f2 = kelly_stake(20000, 0.55, 2.0)
        assert math.isclose(s2, 2.0 * s1, rel_tol=1e-9)
        assert math.isclose(f2, f1, rel_tol=1e-9)

    def test_stake_zero_at_zero_bankroll(self):
        s, f = kelly_stake(0, 0.55, 2.0)
        assert s == 0.0
        assert f == 0.0

    def test_stake_zero_at_negative_bankroll(self):
        s, f = kelly_stake(-100, 0.55, 2.0)
        assert s == 0.0
