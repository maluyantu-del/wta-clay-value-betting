"""Kelly + fractional Kelly position sizing.

Theory
------
For a binary bet at decimal odd ``o`` with true win probability ``p``::

    b = o - 1                # net profit per unit if win
    q = 1 - p                # loss probability
    f* = (b·p - q) / b       # full-Kelly fraction of bankroll

This maximises the expected log-growth of the bankroll. Properties:

  - ``f* > 0`` iff ``p · o > 1`` (positive EV).
  - ``f* = 0`` at zero EV.
  - ``f*`` scales linearly in edge ``(b·p − q)`` and inversely in odd minus 1.

Why fractional Kelly
--------------------
Full-Kelly is mathematically optimal **only** with perfect knowledge of
``p``. Mis-estimation of ``p`` by even 5pp can push full-Kelly into ruin
because variance scales as ``f² × p × q × b²``.

Fractional Kelly (typically 1/4 or 1/2) reduces variance dramatically with
only modest loss of expected log-growth: at 1/4-Kelly you keep ~75% of
the optimal growth rate but ~6% of the variance. Industry standard for
sports models with calibrated-but-imperfect probabilities.

Single-bet cap
--------------
Even fractional Kelly can ask for 8-12% of bankroll on a "stale-line"
edge. Capping at ``max_bet_pct`` (default 2%) prevents one match from
dominating bankroll volatility. This is the same volatility-control idea
as a Sharpe-optimal portfolio with bounded leverage.
"""
from __future__ import annotations

import numpy as np


# Defaults — locked at the project level.
DEFAULT_FRACTION: float = 0.25       # 1/4-Kelly
DEFAULT_MAX_BET_PCT: float = 0.02    # 2% of bankroll cap


def kelly_fraction(prob: float, odds: float) -> float:
    """Standard Kelly fraction. No-short version (returns 0 at negative EV).

    Parameters
    ----------
    prob : float
        Win probability in [0, 1].
    odds : float
        Decimal odds, must be > 1.

    Returns
    -------
    float
        Bankroll fraction to stake. ``0`` if EV ≤ 0 or prob ≥ 1.
    """
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob!r}")
    if not np.isfinite(odds) or odds <= 1.0:
        raise ValueError(f"odds must be finite > 1, got {odds!r}")
    b = float(odds) - 1.0
    p = float(prob)
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def fractional_kelly(
    prob: float,
    odds: float,
    fraction: float = DEFAULT_FRACTION,
    max_bet_pct: float = DEFAULT_MAX_BET_PCT,
) -> float:
    """Fractional Kelly stake with a per-bet cap.

    Parameters
    ----------
    prob, odds
        See :func:`kelly_fraction`.
    fraction : float
        Multiplier on full Kelly. Default ``0.25`` (1/4-Kelly).
    max_bet_pct : float
        Hard cap on bankroll fraction. Default ``0.02`` (2%).

    Returns
    -------
    float
        Bankroll fraction in ``[0, max_bet_pct]``.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    if not (0.0 < max_bet_pct <= 1.0):
        raise ValueError(f"max_bet_pct must be in (0, 1], got {max_bet_pct!r}")
    f_full = kelly_fraction(prob, odds)
    f = f_full * fraction
    return min(f, max_bet_pct)


def kelly_stake(
    bankroll: float,
    prob: float,
    odds: float,
    *,
    fraction: float = DEFAULT_FRACTION,
    max_bet_pct: float = DEFAULT_MAX_BET_PCT,
) -> tuple[float, float]:
    """Compute the absolute stake (currency) and the bankroll fraction used.

    Returns ``(stake, fraction_used)``.
    """
    if bankroll <= 0:
        return 0.0, 0.0
    f = fractional_kelly(prob, odds,
                          fraction=fraction, max_bet_pct=max_bet_pct)
    stake = bankroll * f
    return stake, f
