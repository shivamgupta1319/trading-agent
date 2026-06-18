"""
backtest.py — The CREDIBILITY layer. Does this strategy actually work?

A strategy that "looks smart" is worthless until you've tested it on history. This
module replays a strategy bar-by-bar over past data and measures the truth:
how often did it win, what was the real reward:risk, and is the EXPECTANCY positive
(does it make money on average per trade)?

THE #1 BEGINNER TRAP — look-ahead bias:
  It's terrifyingly easy to accidentally "peek" at information you wouldn't have had
  in real time, which makes a useless strategy look amazing. We avoid it with one
  hard rule: the entry SIGNAL is computed on bar t (today's close), but we ENTER at
  bar t+1's OPEN (tomorrow morning) — you can never act on a candle before it closes.

We simulate ONE position at a time (no pyramiding), walking each trade forward until
its stop or target is hit. If both are touched on the same day, we conservatively
assume the STOP hit first (pessimistic = honest).
"""

from dataclasses import asdict, dataclass

import pandas as pd

from indicators import add_all
from strategies import build_long_trade, get_signal_fn


@dataclass
class Stats:
    """The honest report card for a strategy on one stock."""
    strategy: str
    symbol: str
    period: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float        # of closed trades, how many hit target
    avg_win_r: float           # average winner, in "R" multiples (R = 1 unit of risk)
    avg_loss_r: float          # average loser, in R (usually about -1)
    expectancy_r: float        # average R per trade — THE key number; >0 means edge
    open_trades: int           # trades still running at the end of the data

    def to_dict(self) -> dict:
        return asdict(self)


def _run_backtest(df: pd.DataFrame, signals: pd.Series) -> dict:
    """Core engine: given price data + a per-bar boolean signal, simulate trades.

    Returns the raw trade outcomes as R-multiples. `signals` and `df` share an index.
    R-multiple = profit/loss measured in units of the initial risk (entry - stop).
    A winner that hits a 2:1 target is +2R; a stop-out is -1R.
    """
    df = df.reset_index(drop=True)
    signals = signals.reset_index(drop=True)
    n = len(df)

    r_multiples = []   # closed-trade results, in R
    open_count = 0
    i = 0
    while i < n - 1:
        # Signal fires on bar i (its close), so we may enter on bar i+1's open.
        if not bool(signals.iloc[i]):
            i += 1
            continue

        atr = df["ATR14"].iloc[i]
        entry = float(df["Open"].iloc[i + 1])
        if pd.isna(atr) or pd.isna(entry):
            i += 1
            continue

        stop, target = build_long_trade(entry, float(atr))
        risk = entry - stop
        if risk <= 0:
            i += 1
            continue

        # Walk forward from the entry bar until stop or target is touched.
        exit_idx = None
        result_r = None
        for j in range(i + 1, n):
            low = float(df["Low"].iloc[j])
            high = float(df["High"].iloc[j])
            if low <= stop:                     # stop first (pessimistic tie-break)
                result_r = (stop - entry) / risk    # = -1R
                exit_idx = j
                break
            if high >= target:
                result_r = (target - entry) / risk  # = +2R
                exit_idx = j
                break

        if exit_idx is None:
            # Trade never closed within the data — count it as open, don't score it.
            open_count += 1
            break  # nothing after this can close either; stop scanning

        r_multiples.append(result_r)
        i = exit_idx + 1   # no overlapping trades; resume after this one exits

    return {"r_multiples": r_multiples, "open_count": open_count}


def backtest_strategy(df: pd.DataFrame, symbol: str, strategy: str = "trend", period: str = "2y") -> Stats:
    """Backtest a named strategy on a stock's history and return honest Stats."""
    df = add_all(df)
    signals = get_signal_fn(strategy)(df)
    raw = _run_backtest(df, signals)

    rs = raw["r_multiples"]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    n_closed = len(rs)

    win_rate = (len(wins) / n_closed * 100) if n_closed else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (sum(rs) / n_closed) if n_closed else 0.0

    return Stats(
        strategy=strategy,
        symbol=symbol.upper(),
        period=period,
        trades=n_closed,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=round(win_rate, 1),
        avg_win_r=round(avg_win, 2),
        avg_loss_r=round(avg_loss, 2),
        expectancy_r=round(expectancy, 3),
        open_trades=raw["open_count"],
    )
