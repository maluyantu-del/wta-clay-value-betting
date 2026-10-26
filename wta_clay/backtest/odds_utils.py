"""Bookmaker odds utilities.

Decimal odds conventions
------------------------
A decimal odd ``o`` for an outcome means: bet 1 unit, win ``o`` units back
total (net profit ``o − 1``). Implied probability is ``1 / o``. A pair of
two-way odds (p1 wins, p2 wins) sums implied probabilities to ``1 + vig``,
where ``vig`` is the bookmaker's overround (typically 1-5% on Pinnacle).

Vig-removal methods
-------------------
**Proportional (used here)**: divide each implied probability by the sum.
Simple, interpretable, and the de-facto industry standard for two-way
markets. Other methods (Shin, log-balanced) exist but the differences are
< 0.5pp on a typical vig of 2-5%, well below model edge.
"""
from __future__ import annotations

import numpy as np

#: Minimum allowed decimal odd. Below this is degenerate (or a typo).
_MIN_ODD: float = 1.0


def implied_prob_from_decimal(odd: float) -> float:
    """Implied probability for a decimal odd: ``1 / odd``.

    Parameters
    ----------
    odd : float
        Decimal odd. Must be > 1 for a real betting market (an odd of 1
        implies probability 1, i.e. a sure bet at no payout, which is
        never offered).
    """
    if not np.isfinite(odd) or odd <= _MIN_ODD:
        raise ValueError(f"odd must be a finite number > 1, got {odd!r}")
    return 1.0 / float(odd)


def remove_vig_two_way(p1: float, p2: float) -> tuple[float, float]:
    """Remove bookmaker vig from a two-way market via the proportional method.

    Given implied probabilities ``p1`` and ``p2`` whose sum exceeds 1 (the
    bookmaker overround), returns fair probabilities

        (p1 / (p1 + p2),  p2 / (p1 + p2))

    that sum to exactly 1. If ``p1 + p2 <= 0`` an error is raised.

    Notes
    -----
    The proportional method assumes the bookmaker spreads vig in proportion
    to each side's probability — equivalent to the favorite and underdog
    paying the same percentage commission. This is the simplest model and
    differs from Shin's by < 0.5pp on Pinnacle-typical 2.5% vig.
    """
    if not (np.isfinite(p1) and np.isfinite(p2)):
        raise ValueError(f"probs must be finite, got p1={p1!r}, p2={p2!r}")
    s = p1 + p2
    if s <= 0:
        raise ValueError(
            f"sum of implied probs must be > 0, got p1+p2 = {s!r}"
        )
    return float(p1) / float(s), float(p2) / float(s)


def fair_probs_from_decimal(odd_p1: float, odd_p2: float) -> tuple[float, float]:
    """Convenience: decimal odds → vig-removed fair probabilities."""
    p1 = implied_prob_from_decimal(odd_p1)
    p2 = implied_prob_from_decimal(odd_p2)
    return remove_vig_two_way(p1, p2)


def implied_vig(odd_p1: float, odd_p2: float) -> float:
    """Compute the bookmaker overround: ``1/odd_p1 + 1/odd_p2 - 1``."""
    return (implied_prob_from_decimal(odd_p1)
            + implied_prob_from_decimal(odd_p2) - 1.0)


def ev_from_prob_and_odd(prob: float, odd: float) -> float:
    """Expected value of a 1-unit bet at ``odd`` given true win prob ``prob``.

    EV = prob * (odd - 1) - (1 - prob)
       = prob * odd - 1
    """
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob!r}")
    if not np.isfinite(odd) or odd <= _MIN_ODD:
        raise ValueError(f"odd must be finite > 1, got {odd!r}")
    return float(prob) * float(odd) - 1.0
