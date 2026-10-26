"""Bankroll-aware backtest engine.

Walks a sorted-by-date stream of test predictions, places bets when the
calibrated EV exceeds threshold, sizes via fractional Kelly with a
single-bet cap, and tracks bankroll trajectory.

Inputs and outputs
------------------
``run_backtest`` takes a *predictions* DataFrame and the master DataFrame
(for date / odds / tier lookups). Returns a :class:`BacktestResult` with
trades, equity curve, summary, and monthly PnL.

Filter rules (mirrors mini-backtest)
------------------------------------
- ``not_for_betting=True`` rows excluded
- ``market_source='none'`` rows excluded
- 2009 rows excluded (Pinnacle gap, only relevant if test happens to span 2009)
- Rows with NaN selected-book odds excluded

Bet logic
---------
For each row, compute EV both sides::

    ev_p1 = p_model       · (odds_p1 − 1) − (1 − p_model)
    ev_p2 = (1 − p_model) · (odds_p2 − 1) − p_model

Place a bet on whichever side has ``ev > ev_threshold``. If both sides
exceed (rare, requires negative vig), pick the higher-EV side. Stake via
:func:`wta_clay.backtest.staking.fractional_kelly` against the current
bankroll, capped at ``max_bet_pct``. Update bankroll after each bet.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from wta_clay.backtest.metrics import (
    annualized_return_pct, avg_ev, avg_odds_taken, avg_pnl_per_bet,
    avg_stake, daily_returns, hit_rate, kelly_concentration,
    max_drawdown_pct, n_bets, n_wins, roi_per_bet, sharpe,
    total_return_pct,
)
from wta_clay.backtest.staking import (
    DEFAULT_FRACTION, DEFAULT_MAX_BET_PCT, fractional_kelly,
)

log = logging.getLogger(__name__)


BookChoice = Literal["pinnacle", "b365", "best"]


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`.

    Attributes
    ----------
    trades : DataFrame
        One row per placed bet. Columns:
          match_id, match_date, p1_id, p2_id, p1_name, p2_name,
          tour_level, model_prob, market_prob, ev,
          bet_side, odds_taken, kelly_frac, stake,
          y_true, won, pnl, bankroll_after
    equity_curve : Series
        Bankroll values indexed by match_date (one entry per processed
        match — bets AND skipped rows). Useful for time-of-day drawdowns.
    summary : dict
        Headline numbers (initial / final bankroll, ROI, CAGR, MDD,
        Sharpe, n_bets, hit rate, etc.).
    monthly_pnl : DataFrame
        Per (year, month) PnL aggregated.
    config : dict
        Run-time configuration echoed back.
    """
    trades: pd.DataFrame
    equity_curve: pd.Series
    summary: dict
    monthly_pnl: pd.DataFrame
    config: dict = field(default_factory=dict)


# ===========================================================================
# Main entry
# ===========================================================================

def run_backtest(
    predictions: pd.DataFrame,
    master: pd.DataFrame,
    *,
    initial_bankroll: float = 10000.0,
    ev_threshold: float = 0.05,
    kelly_fraction_arg: float = DEFAULT_FRACTION,
    max_bet_pct: float = DEFAULT_MAX_BET_PCT,
    book: BookChoice = "pinnacle",
) -> BacktestResult:
    """Run a bankroll-aware backtest on calibrated test predictions.

    Parameters
    ----------
    predictions : DataFrame
        Must contain columns ``match_id`` and ``y_pred_prob`` (calibrated).
        Optionally ``y_true``. Other rows from master are looked up by id.
    master : DataFrame
        Master DataFrame; used to look up odds, dates, tier, names.
    initial_bankroll : float
        Starting bankroll. Default 10000 (currency units).
    ev_threshold : float
        Minimum EV to place a bet. Default 5%.
    kelly_fraction_arg : float
        Fractional-Kelly multiplier on full Kelly. Default 0.25 (1/4).
    max_bet_pct : float
        Hard cap on bet size as fraction of bankroll. Default 0.02 (2%).
    book : str
        Which odds to use: ``"pinnacle"``, ``"b365"``, or ``"best"``
        (max of the two).
    """
    if "match_id" not in predictions.columns or "y_pred_prob" not in predictions.columns:
        raise ValueError(
            "predictions must contain 'match_id' and 'y_pred_prob' columns"
        )

    config = {
        "initial_bankroll": float(initial_bankroll),
        "ev_threshold":     float(ev_threshold),
        "kelly_fraction":   float(kelly_fraction_arg),
        "max_bet_pct":      float(max_bet_pct),
        "book":             str(book),
    }

    # ------------------------------------------------------------------
    # 1. Build the per-match working frame: predictions + master metadata
    # ------------------------------------------------------------------
    extra_cols = [
        "match_id", "match_date", "tour_level",
        "p1_id", "p2_id", "p1_name", "p2_name",
        "odds_p1_pinnacle", "odds_p2_pinnacle",
        "odds_p1_b365", "odds_p2_b365",
        "not_for_betting", "market_source",
    ]
    extra_cols = [c for c in extra_cols if c in master.columns]
    df = predictions.merge(master[extra_cols], on="match_id", how="left",
                            suffixes=("", "_m"))

    # Required outcome col
    if "y_true" not in df.columns and "y" in master.columns:
        df = df.merge(master[["match_id", "y"]].rename(columns={"y": "y_true"}),
                       on="match_id", how="left")

    # ------------------------------------------------------------------
    # 2. Pick odds based on `book`
    # ------------------------------------------------------------------
    df["odds_p1"], df["odds_p2"] = _pick_odds(df, book)

    # ------------------------------------------------------------------
    # 3. Apply skip filters (record skipped reason for diagnostics)
    # ------------------------------------------------------------------
    keep = pd.Series(True, index=df.index)
    if "not_for_betting" in df.columns:
        keep &= ~df["not_for_betting"].fillna(False).astype(bool)
    if "market_source" in df.columns:
        keep &= df["market_source"].astype(str) != "none"
    if "match_date" in df.columns:
        keep &= df["match_date"].dt.year != 2009
    keep &= df["odds_p1"].notna() & df["odds_p2"].notna()

    df_eligible = df.loc[keep].sort_values("match_date").reset_index(drop=True)
    log.info("backtest eligible: %d / %d rows kept", len(df_eligible), len(df))

    # ------------------------------------------------------------------
    # 4. Walk chronologically, place bets, update bankroll
    # ------------------------------------------------------------------
    bankroll = float(initial_bankroll)
    trades_rows: list[dict] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []

    for _, row in df_eligible.iterrows():
        p_model = float(row["y_pred_prob"])
        odds_p1 = float(row["odds_p1"])
        odds_p2 = float(row["odds_p2"])
        y_true  = int(row["y_true"]) if pd.notna(row.get("y_true")) else None

        ev_p1 = p_model * (odds_p1 - 1.0) - (1.0 - p_model)
        ev_p2 = (1.0 - p_model) * (odds_p2 - 1.0) - p_model

        # Pick the side(s) with ev > threshold
        bet_side: str | None = None
        bet_prob = None
        bet_odds = None
        bet_ev = None
        if ev_p1 > ev_threshold and ev_p2 > ev_threshold:
            if ev_p1 >= ev_p2:
                bet_side, bet_prob, bet_odds, bet_ev = "p1", p_model, odds_p1, ev_p1
            else:
                bet_side, bet_prob, bet_odds, bet_ev = "p2", 1 - p_model, odds_p2, ev_p2
        elif ev_p1 > ev_threshold:
            bet_side, bet_prob, bet_odds, bet_ev = "p1", p_model, odds_p1, ev_p1
        elif ev_p2 > ev_threshold:
            bet_side, bet_prob, bet_odds, bet_ev = "p2", 1 - p_model, odds_p2, ev_p2

        if bet_side is not None and bankroll > 0:
            f = fractional_kelly(
                bet_prob, bet_odds,
                fraction=kelly_fraction_arg,
                max_bet_pct=max_bet_pct,
            )
            stake = bankroll * f
            if stake > 0:
                # Resolve PnL
                if bet_side == "p1":
                    won = (y_true == 1) if y_true is not None else False
                else:
                    won = (y_true == 0) if y_true is not None else False
                pnl = stake * (bet_odds - 1.0) if won else -stake
                bankroll += pnl

                # Vig-removed market_prob for diagnostic
                imp1 = 1.0 / odds_p1
                imp2 = 1.0 / odds_p2
                s = imp1 + imp2
                market_prob_p1 = imp1 / s if s > 0 else float("nan")
                market_prob = (market_prob_p1 if bet_side == "p1"
                               else 1.0 - market_prob_p1)

                trades_rows.append({
                    "match_id":      int(row["match_id"]),
                    "match_date":    row["match_date"],
                    "tour_level":    str(row.get("tour_level", "")),
                    "p1_id":         int(row["p1_id"]) if pd.notna(row.get("p1_id")) else None,
                    "p2_id":         int(row["p2_id"]) if pd.notna(row.get("p2_id")) else None,
                    "p1_name":       str(row.get("p1_name", "")),
                    "p2_name":       str(row.get("p2_name", "")),
                    "model_prob":    p_model,
                    "market_prob":   market_prob,
                    "ev":            bet_ev,
                    "bet_side":      bet_side,
                    "odds_taken":    bet_odds,
                    "kelly_frac":    f,
                    "stake":         stake,
                    "y_true":        y_true,
                    "won":           bool(won),
                    "pnl":           pnl,
                    "bankroll_after": bankroll,
                })

        equity_records.append((row["match_date"], bankroll))

    # ------------------------------------------------------------------
    # 5. Build outputs
    # ------------------------------------------------------------------
    trades = pd.DataFrame(trades_rows)
    if not trades.empty:
        trades["match_date"] = pd.to_datetime(trades["match_date"])

    if equity_records:
        eq_idx = pd.DatetimeIndex([d for d, _ in equity_records])
        equity = pd.Series([b for _, b in equity_records], index=eq_idx,
                            name="bankroll")
        equity = equity.sort_index()
    else:
        equity = pd.Series(dtype=float, name="bankroll")

    # Time span for annualisation
    if not trades.empty:
        n_years = max(
            (trades["match_date"].max() - trades["match_date"].min()).days / 365.25,
            1.0 / 365.25,
        )
    else:
        n_years = float("nan")

    monthly = _build_monthly_pnl(trades)

    summary = {
        "initial_bankroll":  initial_bankroll,
        "final_bankroll":    float(bankroll),
        "n_eligible":        int(len(df_eligible)),
        "n_bets":            n_bets(trades),
        "n_wins":            n_wins(trades),
        "hit_rate":          hit_rate(trades),
        "roi_per_bet":       roi_per_bet(trades),
        "total_return_pct":  total_return_pct(equity),
        "annualized_return_pct":
            annualized_return_pct(equity, n_years) if not np.isnan(n_years) else float("nan"),
        "max_drawdown_pct":  max_drawdown_pct(equity),
        "sharpe":            sharpe(daily_returns(trades, equity)),
        "avg_odds":          avg_odds_taken(trades),
        "avg_ev":            avg_ev(trades),
        "avg_stake":         avg_stake(trades),
        "avg_pnl_per_bet":   avg_pnl_per_bet(trades),
        "kelly_concentration": kelly_concentration(trades),
        "n_years":           n_years,
        "book":              book,
    }
    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        summary=summary,
        monthly_pnl=monthly,
        config=config,
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _pick_odds(df: pd.DataFrame, book: BookChoice) -> tuple[pd.Series, pd.Series]:
    """Return (odds_p1, odds_p2) Series per the `book` setting."""
    if book == "pinnacle":
        return df.get("odds_p1_pinnacle"), df.get("odds_p2_pinnacle")
    if book == "b365":
        return df.get("odds_p1_b365"), df.get("odds_p2_b365")
    if book == "best":
        # Element-wise max ignoring NaN; both columns must exist
        if "odds_p1_b365" not in df.columns:
            return df.get("odds_p1_pinnacle"), df.get("odds_p2_pinnacle")
        p1 = df["odds_p1_pinnacle"].fillna(0).combine(
            df["odds_p1_b365"].fillna(0), max)
        p2 = df["odds_p2_pinnacle"].fillna(0).combine(
            df["odds_p2_b365"].fillna(0), max)
        # Restore NaN if both books had NaN (combined max is 0)
        p1 = p1.where(p1 > 0, other=np.nan)
        p2 = p2.where(p2 > 0, other=np.nan)
        return p1, p2
    raise ValueError(f"unknown book {book!r}")


def _build_monthly_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PnL by (year, month).

    Returns a wide table:
        year, month, n_bets, pnl, stake, roi
    """
    if trades.empty:
        return pd.DataFrame(columns=["year","month","n_bets","pnl","stake","roi"])
    t = trades.copy()
    t["year"] = t["match_date"].dt.year
    t["month"] = t["match_date"].dt.month
    g = t.groupby(["year", "month"])
    out = g.agg(
        n_bets=("pnl", "size"),
        pnl=("pnl", "sum"),
        stake=("stake", "sum"),
    ).reset_index()
    out["roi"] = out["pnl"] / out["stake"]
    return out.sort_values(["year", "month"]).reset_index(drop=True)
