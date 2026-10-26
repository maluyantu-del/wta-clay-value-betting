"""Grouped reports + equity-curve plot.

All grouping helpers take a ``trades`` DataFrame (the output of
:func:`wta_clay.backtest.engine.run_backtest`) and return a wide DataFrame
keyed by the grouping dimension. The plot helper saves a PNG of the
equity curve for visual sanity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLAY_SEASON_MONTHS: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)


# ===========================================================================
# Grouping primitive
# ===========================================================================

def _group_summary(g: pd.DataFrame, *, by_label: str) -> dict:
    """Return aggregate stats for a sub-DataFrame of trades."""
    if g.empty:
        return {by_label: None, "n_bets": 0, "n_wins": 0,
                "hit_rate_%": None, "stake": 0.0, "pnl": 0.0,
                "roi_%": None, "avg_odds": None, "avg_ev": None}
    n = int(len(g))
    nw = int(g["won"].sum())
    s = float(g["stake"].sum())
    p = float(g["pnl"].sum())
    return {
        "n_bets":    n,
        "n_wins":    nw,
        "hit_rate_%": round(nw / n * 100, 2),
        "stake":     round(s, 2),
        "pnl":       round(p, 2),
        "roi_%":     round(p / s * 100, 2) if s > 0 else None,
        "avg_odds":  round(float(g["odds_taken"].mean()), 3),
        "avg_ev":    round(float(g["ev"].mean()), 4),
    }


# ===========================================================================
# Per-dimension reports
# ===========================================================================

def report_by_tourney_level(trades: pd.DataFrame) -> pd.DataFrame:
    """ROI / hit-rate / count grouped by ``tour_level``.

    Output ordered: GS, WTA1000, WTA500, WTA250, WTA125 (canonical).
    """
    cols_order = ["GS", "WTA1000", "WTA500", "WTA250", "WTA125", "ITF"]
    if trades.empty or "tour_level" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for level in cols_order:
        g = trades[trades["tour_level"] == level]
        s = _group_summary(g, by_label="tour_level")
        s["tour_level"] = level
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[["tour_level"] + [c for c in df.columns if c != "tour_level"]]


def report_by_odds_bucket(
    trades: pd.DataFrame,
    bins: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf),
) -> pd.DataFrame:
    """ROI by decimal-odds bucket (book-line bands).

    Default bands: 1.0-1.5 (heavy fav) / 1.5-2.0 / 2.0-3.0 / 3.0-5.0 / 5.0-10.0 / 10+
    """
    if trades.empty:
        return pd.DataFrame()
    labels = [f"{bins[i]:g}-{bins[i+1]:g}" for i in range(len(bins) - 1)]
    cuts = pd.cut(trades["odds_taken"], bins=bins, right=False, labels=labels,
                   include_lowest=True)
    rows = []
    for lbl in labels:
        g = trades[cuts == lbl]
        s = _group_summary(g, by_label="odds_bucket")
        s["odds_bucket"] = lbl
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[["odds_bucket"] + [c for c in df.columns if c != "odds_bucket"]]


def report_by_ev_bucket(
    trades: pd.DataFrame,
    bins: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 1.0),
) -> pd.DataFrame:
    """ROI by EV bucket (5-10%, 10-15%, 15-25%, 25%+)."""
    if trades.empty:
        return pd.DataFrame()
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%"
              for i in range(len(bins) - 1)]
    cuts = pd.cut(trades["ev"], bins=bins, right=False, labels=labels,
                   include_lowest=True)
    rows = []
    for lbl in labels:
        g = trades[cuts == lbl]
        s = _group_summary(g, by_label="ev_bucket")
        s["ev_bucket"] = lbl
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[["ev_bucket"] + [c for c in df.columns if c != "ev_bucket"]]


def report_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    """ROI per calendar year."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["year"] = t["match_date"].dt.year
    rows = []
    for yr, g in t.groupby("year"):
        s = _group_summary(g, by_label="year")
        s["year"] = int(yr)
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[["year"] + [c for c in df.columns if c != "year"]].sort_values("year")


def report_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    """ROI per calendar month — useful to surface clay-season seasonality."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["month"] = t["match_date"].dt.month
    t["clay_season"] = t["month"].isin(CLAY_SEASON_MONTHS)
    rows = []
    for m in range(1, 13):
        g = t[t["month"] == m]
        s = _group_summary(g, by_label="month")
        s["month"] = m
        s["clay_season"] = m in CLAY_SEASON_MONTHS
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[["month","clay_season"]
              + [c for c in df.columns if c not in ("month","clay_season")]]


def report_clay_vs_offseason(trades: pd.DataFrame) -> pd.DataFrame:
    """Two-row summary: clay-season (Apr-Oct) vs off-season."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["clay_season"] = t["match_date"].dt.month.isin(CLAY_SEASON_MONTHS)
    rows = []
    for is_clay in (True, False):
        g = t[t["clay_season"] == is_clay]
        s = _group_summary(g, by_label="period")
        s["period"] = "clay_season (Apr-Oct)" if is_clay else "off_season"
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[["period"] + [c for c in df.columns if c != "period"]]


# ===========================================================================
# Equity curve plot
# ===========================================================================

def plot_equity_curve(
    equity: pd.Series,
    out_path: str | Path,
    *,
    title: str = "Backtest equity curve",
    initial_bankroll: float | None = None,
) -> Path:
    """Save a PNG of the bankroll trajectory.

    Falls back gracefully if matplotlib is unavailable (returns the path
    untouched and logs a warning).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:    # pragma: no cover
        log.warning("matplotlib unavailable; skipping equity-curve plot")
        return out_path

    if equity.empty:
        log.warning("equity curve is empty; skipping plot")
        return out_path

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, linewidth=1.4, color="#1f77b4")
    if initial_bankroll is not None:
        ax.axhline(initial_bankroll, color="#888888", linestyle="--",
                    linewidth=0.8, alpha=0.7,
                    label=f"start = {initial_bankroll:,.0f}")
    ax.set_title(title)
    ax.set_xlabel("Match date")
    ax.set_ylabel("Bankroll")
    ax.grid(True, alpha=0.3)
    if initial_bankroll is not None:
        ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info("equity curve saved to %s", out_path)
    return out_path


# ===========================================================================
# Pretty-print helpers
# ===========================================================================

def format_summary(summary: dict) -> str:
    """Multi-line text summary of the backtest's headline numbers."""
    lines: list[str] = []
    lines.append(f"  Initial bankroll:       {summary['initial_bankroll']:>14,.2f}")
    lines.append(f"  Final bankroll:         {summary['final_bankroll']:>14,.2f}")
    lines.append(f"  Total return:           {summary['total_return_pct']:>13.2f}%")
    lines.append(f"  Annualized return:      {summary['annualized_return_pct']:>13.2f}%")
    lines.append(f"  Max drawdown:           {summary['max_drawdown_pct']:>13.2f}%")
    lines.append(f"  Sharpe ratio:           {summary['sharpe']:>14.3f}")
    lines.append(f"  Span (years):           {summary['n_years']:>14.2f}")
    lines.append("")
    lines.append(f"  Eligible rows:          {summary['n_eligible']:>14d}")
    lines.append(f"  N bets:                 {summary['n_bets']:>14d}")
    lines.append(f"  N wins:                 {summary['n_wins']:>14d}")
    lines.append(f"  Hit rate:               {(summary['hit_rate'] or 0)*100:>13.2f}%")
    lines.append(f"  ROI per bet:            {(summary['roi_per_bet'] or 0)*100:>13.2f}%")
    lines.append(f"  Avg odds taken:         {summary['avg_odds']:>14.3f}")
    lines.append(f"  Avg EV:                 {(summary['avg_ev'] or 0)*100:>13.2f}%")
    lines.append(f"  Avg stake:              {summary['avg_stake']:>14,.2f}")
    lines.append(f"  Avg PnL/bet:            {summary['avg_pnl_per_bet']:>14,.2f}")
    kc = summary["kelly_concentration"]
    lines.append(f"  Kelly frac mean:        {kc['mean']:>14.4f}")
    lines.append(f"  Kelly frac max:         {kc['max']:>14.4f}")
    lines.append(f"  Book:                   {summary['book']:>14}")
    return "\n".join(lines)
