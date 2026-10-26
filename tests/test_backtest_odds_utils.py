"""Unit tests for ``wta_clay.backtest.odds_utils``."""
from __future__ import annotations

import math

import pytest

from wta_clay.backtest.odds_utils import (
    ev_from_prob_and_odd,
    fair_probs_from_decimal,
    implied_prob_from_decimal,
    implied_vig,
    remove_vig_two_way,
)


class TestImpliedProb:
    def test_simple(self):
        assert math.isclose(implied_prob_from_decimal(2.0), 0.5)
        assert math.isclose(implied_prob_from_decimal(4.0), 0.25)

    def test_extreme(self):
        assert math.isclose(implied_prob_from_decimal(1.01), 1 / 1.01,
                              rel_tol=1e-9)

    def test_invalid_low(self):
        with pytest.raises(ValueError):
            implied_prob_from_decimal(1.0)
        with pytest.raises(ValueError):
            implied_prob_from_decimal(0.5)

    def test_invalid_nan(self):
        with pytest.raises(ValueError):
            implied_prob_from_decimal(float("nan"))


class TestRemoveVigTwoWay:
    def test_no_vig_returns_input_normalised(self):
        # 0.5/0.5 sums to 1, already fair
        p1, p2 = remove_vig_two_way(0.5, 0.5)
        assert math.isclose(p1, 0.5)
        assert math.isclose(p2, 0.5)

    def test_typical_vig(self):
        # Pinnacle 1.95/2.10 → implied 0.513 + 0.476 = 0.989  (sub-1, weird but
        # math still works). Use cleaner case: 1.85/2.10 → 0.541 + 0.476 = 1.017
        p1_imp = 1 / 1.85
        p2_imp = 1 / 2.10
        p1, p2 = remove_vig_two_way(p1_imp, p2_imp)
        assert math.isclose(p1 + p2, 1.0, abs_tol=1e-9)
        assert p1 > p2   # 1.85 favorite

    def test_proportional_property(self):
        # Vig-removal must preserve relative ratio
        p1_imp = 0.6
        p2_imp = 0.5
        p1, p2 = remove_vig_two_way(p1_imp, p2_imp)
        # Ratio preserved:
        assert math.isclose(p1 / p2, p1_imp / p2_imp, rel_tol=1e-9)

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            remove_vig_two_way(0.0, 0.0)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            remove_vig_two_way(float("nan"), 0.5)


class TestFairProbs:
    def test_round_trip(self):
        # Decimal odds → fair probs, then back via 1/p should give odds back
        # ONLY in the no-vig case
        p1, p2 = fair_probs_from_decimal(2.0, 2.0)
        assert math.isclose(p1, 0.5)
        assert math.isclose(p2, 0.5)

    def test_with_vig(self):
        p1, p2 = fair_probs_from_decimal(1.85, 2.10)
        assert math.isclose(p1 + p2, 1.0, abs_tol=1e-9)
        assert p1 > 0.5   # favorite


class TestImpliedVig:
    def test_zero_vig(self):
        # 2.0/2.0 → exactly fair
        assert math.isclose(implied_vig(2.0, 2.0), 0.0)

    def test_typical(self):
        # 1.85/2.10 → vig ≈ 1.7%
        v = implied_vig(1.85, 2.10)
        assert 0.01 < v < 0.03

    def test_arbitrage_negative_vig(self):
        # 52/2.77 → implied 0.019 + 0.361 = 0.380, vig = -0.62 (impossible)
        v = implied_vig(52.0, 2.77)
        assert v < -0.5


class TestEvFromProbAndOdd:
    def test_zero_ev_at_fair(self):
        # prob=0.5, odd=2.0 → EV = 1.0 - 1 = 0
        assert math.isclose(ev_from_prob_and_odd(0.5, 2.0), 0.0)

    def test_positive_ev(self):
        # prob=0.55, odd=2.0 → EV = 1.10 - 1 = 0.10
        assert math.isclose(ev_from_prob_and_odd(0.55, 2.0), 0.10)

    def test_negative_ev(self):
        # prob=0.45, odd=2.0 → EV = 0.90 - 1 = -0.10
        assert math.isclose(ev_from_prob_and_odd(0.45, 2.0), -0.10)

    def test_invalid_prob(self):
        with pytest.raises(ValueError):
            ev_from_prob_and_odd(1.5, 2.0)
        with pytest.raises(ValueError):
            ev_from_prob_and_odd(-0.1, 2.0)
