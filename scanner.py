"""
scanner.py — "Show me the best setups in the market right now."

This ties everything together. For a chosen strategy it:
  1. Batch-downloads price data for the whole Nifty 50.
  2. Runs the LIVE strategy on each stock — keeping only those triggering TODAY.
  3. Backtests each triggered stock so we know that signal's historical quality.
  4. RANKS the survivors by backtested expectancy (best edge first).

So a result isn't just "this triggered today" — it's "this triggered today AND this
strategy has a real historical edge on this stock." That ranking is the whole value.
"""

from backtest import backtest_strategy
from data import get_many_ohlcv, normalize_symbol
from strategies import run_strategy
from universe import NIFTY_50, display_name

# A backtest on very few trades is basically noise. Below this, we flag low confidence
# rather than trusting the number.
MIN_TRADES_FOR_CONFIDENCE = 8


def scan(strategy: str = "trend", period: str = "3y", top_n: int = 10, universe=None) -> dict:
    """Scan a universe for triggered setups, ranked by historical edge."""
    universe = universe or NIFTY_50

    data = get_many_ohlcv(universe, period=period)

    # Be transparent about any stocks we couldn't fetch (delisted/renamed/data gaps)
    # rather than silently scanning fewer than advertised.
    fetched = set(data.keys())
    missing = [display_name(normalize_symbol(s)) for s in universe if normalize_symbol(s) not in fetched]

    triggered = []
    for sym, df in data.items():
        if len(df) < 120:   # need enough history for indicators + a meaningful backtest
            continue

        name = display_name(sym)
        setup = run_strategy(df, name, strategy)
        if not setup.triggered:
            continue

        stats = backtest_strategy(df, name, strategy, period=period)
        triggered.append(
            {
                "symbol": name,
                "entry": setup.entry,
                "stop": setup.stop,
                "target": setup.target,
                "risk_reward": setup.risk_reward,
                "backtest_trades": stats.trades,
                "win_rate_pct": stats.win_rate_pct,
                "expectancy_r": stats.expectancy_r,
                "confidence": "ok" if stats.trades >= MIN_TRADES_FOR_CONFIDENCE else "low (few backtest trades)",
            }
        )

    # Rank: best historical edge first; break ties by sample size (more = more trustworthy).
    triggered.sort(key=lambda r: (r["expectancy_r"], r["backtest_trades"]), reverse=True)

    return {
        "strategy": strategy,
        "period": period,
        "scanned": len(universe),
        "with_data": len(data),
        "missing_symbols": missing,   # surfaced, not hidden
        "triggered_count": len(triggered),
        "top": triggered[:top_n],
    }
