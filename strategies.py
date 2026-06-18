"""
strategies.py — Where indicators become a DECISION.

A "strategy" is a set of rules that looks at the indicators and either says
"yes, this is a setup — here's the entry, stop and target" or "no setup, here's why."
We return that as a `Setup` either way, so the agent can always explain itself.

IMPORTANT design point (why this file is structured the way it is):
  The SAME entry rule is used in two places — the live setup (latest bar) AND the
  backtest (every historical bar). If those two ever used different code, your
  backtest would be testing a different strategy than the one you trade — a classic,
  silent, expensive bug. So each strategy's rule lives in ONE vectorized function
  (e.g. `trend_signal`) that returns a True/False for every bar, and both the live
  Setup and the backtester read from it.

Everything is LONG-ONLY (buy expecting a rise). Short-selling comes later.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd

from indicators import add_all
from risk import atr_stop, risk_reward, target_from_rr


# How every long trade is shaped from an entry + volatility. Shared by live & backtest
# so the trade you see is the trade that was tested.
ATR_MULT = 2.0      # stop = entry - 2*ATR
RR_TARGET = 2.0     # target gives 2:1 reward:risk


def build_long_trade(entry: float, atr: float):
    """Given an entry price and ATR, return (stop, target) for a long trade."""
    stop = atr_stop(entry, atr, atr_mult=ATR_MULT)
    target = target_from_rr(entry, stop, rr_target=RR_TARGET)
    return stop, target


@dataclass
class Setup:
    """A single, fully-specified trade idea (or a 'no trade' with reasons)."""
    symbol: str
    strategy: str
    direction: str            # "long" for now
    triggered: bool           # is the setup active on the latest bar?
    price: float              # latest close, for reference
    reasons: list = field(default_factory=list)  # plain-English why / why-not
    # Filled in only when triggered:
    entry: float = None
    stop: float = None
    target: float = None
    risk_reward: float = None
    atr: float = None

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Strategy 1: Trend-following swing (long)
# ===========================================================================
# Philosophy: "the trend is your friend." Buy stocks already climbing, with
# momentum confirming, but not so overheated we're buying the top. Three rules:
#   1. Uptrend:      close > EMA20 > EMA50
#   2. Healthy RSI:  50 <= RSI <= 70   (momentum present, not overbought)
#   3. MACD confirm: MACD > MACD signal
# ---------------------------------------------------------------------------

def trend_signal(df: pd.DataFrame) -> pd.Series:
    """Vectorized entry rule: a True/False for EVERY bar in df.

    `df` must already have indicator columns (call add_all first). This is the
    single source of truth for the trend strategy — both live and backtest use it.
    """
    uptrend = (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"])
    rsi_ok = df["RSI14"].between(50, 70)
    macd_ok = df["MACD"] > df["MACD_signal"]
    return uptrend & rsi_ok & macd_ok


def _trend_reasons(last) -> list:
    """Human-readable pass/fail for the latest bar (for the live Setup)."""
    close, ema20, ema50 = float(last["Close"]), float(last["EMA20"]), float(last["EMA50"])
    rsi = float(last["RSI14"])
    macd, sig = float(last["MACD"]), float(last["MACD_signal"])

    cond_trend = close > ema20 > ema50
    cond_rsi = 50 <= rsi <= 70
    cond_macd = macd > sig
    return [
        f"{'✓' if cond_trend else '✗'} Uptrend: close {close:.1f} "
        f"{'>' if close > ema20 else '<='} EMA20 {ema20:.1f} "
        f"{'>' if ema20 > ema50 else '<='} EMA50 {ema50:.1f}",
        f"{'✓' if cond_rsi else '✗'} RSI healthy: {rsi:.1f} "
        f"({'in' if cond_rsi else 'outside'} the 50–70 momentum zone)",
        f"{'✓' if cond_macd else '✗'} MACD confirms: {macd:.2f} "
        f"{'>' if cond_macd else '<='} signal {sig:.2f}",
    ]


# ===========================================================================
# Strategy 2: Breakout swing (long)
# ===========================================================================
# Philosophy: "buy strength." When price pushes above its recent ceiling on heavy
# volume while a real trend exists, momentum often continues. Three rules:
#   1. Breakout:  close > highest high of the prior 20 days (a new 20-day high)
#   2. Volume:    relative volume > 1.2 (the move has real participation, not a fluke)
#   3. Strength:  ADX > 20 (there's an actual trend, not just noise)
# ---------------------------------------------------------------------------
BREAKOUT_LOOKBACK = 20


def breakout_signal(df: pd.DataFrame) -> pd.Series:
    # Prior 20-day high, EXCLUDING today (.shift(1)) so we compare against the past,
    # not a window that already contains today's bar.
    prior_high = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    broke_out = df["Close"] > prior_high
    vol_ok = df["RelVolume"] > 1.2
    adx_ok = df["ADX14"] > 20
    return (broke_out & vol_ok & adx_ok).fillna(False)


def _breakout_reasons(df_last_two) -> list:
    """Takes the indicator df (needs prior bars for the rolling high)."""
    last = df_last_two.iloc[-1]
    prior_high = float(df_last_two["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
    close = float(last["Close"])
    rel_vol = float(last["RelVolume"]) if not pd.isna(last["RelVolume"]) else 0.0
    adx = float(last["ADX14"]) if not pd.isna(last["ADX14"]) else 0.0

    c1, c2, c3 = close > prior_high, rel_vol > 1.2, adx > 20
    return [
        f"{'✓' if c1 else '✗'} Breakout: close {close:.1f} "
        f"{'>' if c1 else '<='} 20-day high {prior_high:.1f}",
        f"{'✓' if c2 else '✗'} Volume: {rel_vol:.2f}x average "
        f"({'above' if c2 else 'below'} the 1.2x threshold)",
        f"{'✓' if c3 else '✗'} Trend strength: ADX {adx:.1f} "
        f"({'>' if c3 else '<='} 20)",
    ]


# ===========================================================================
# Strategy 3: Mean-reversion dip-buy (long)
# ===========================================================================
# Philosophy: "buy the dip in an uptrend." In a stock that's structurally rising,
# a short-term pullback below its 20-day average with a weak RSI often snaps back.
# The opposite of breakout — we buy weakness, not strength. Three rules:
#   1. Bigger uptrend: close > EMA50 (only buy dips in things trending up)
#   2. Weak short-term: RSI < 45 (a real pullback, below the 50 midline)
#   3. Below the mean:  close < BB middle band (price has dipped below its 20-day avg)
# (Two lessons are baked into these thresholds: (a) "below the LOWER band" almost
#  never fires on large-caps — too selective to be useful; (b) we measured that
#  RSI<40 AND uptrend co-occur on ~13 of 12,000 bars — once RSI is that low price has
#  usually already lost the EMA50. RSI<45 keeps the spirit while actually trading.
#  Moral: pick thresholds from data, and let the backtest judge them.)
# ---------------------------------------------------------------------------
def mean_reversion_signal(df: pd.DataFrame) -> pd.Series:
    uptrend = df["Close"] > df["EMA50"]
    weak = df["RSI14"] < 45
    below_mean = df["Close"] < df["BB_mid"]
    return (uptrend & weak & below_mean).fillna(False)


def _mean_reversion_reasons(last) -> list:
    close, ema50 = float(last["Close"]), float(last["EMA50"])
    rsi = float(last["RSI14"])
    bb_mid = float(last["BB_mid"]) if not pd.isna(last["BB_mid"]) else float("nan")

    c1, c2, c3 = close > ema50, rsi < 45, close < bb_mid
    return [
        f"{'✓' if c1 else '✗'} Bigger uptrend: close {close:.1f} "
        f"{'>' if c1 else '<='} EMA50 {ema50:.1f}",
        f"{'✓' if c2 else '✗'} Pullback: RSI {rsi:.1f} "
        f"({'<' if c2 else '>='} 45)",
        f"{'✓' if c3 else '✗'} Below 20-day mean: close {close:.1f} "
        f"{'<' if c3 else '>='} mid band {bb_mid:.1f}",
    ]


# ---------------------------------------------------------------------------
# Strategy registry. Each entry knows how to (a) generate per-bar signals and
# (b) describe the latest bar's reasons. Adding a strategy = one new block of
# functions + one line here. (Some reason-builders need the whole df, not just
# the last row — they take `needs_df: True`.)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "trend": {"signal": trend_signal, "reasons": _trend_reasons, "needs_df": False},
    "breakout": {"signal": breakout_signal, "reasons": _breakout_reasons, "needs_df": True},
    "mean_reversion": {"signal": mean_reversion_signal, "reasons": _mean_reversion_reasons, "needs_df": False},
}


def get_signal_fn(strategy: str):
    """Return the vectorized signal function for a strategy name (used by backtest)."""
    spec = STRATEGIES.get(strategy)
    if spec is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(STRATEGIES)}")
    return spec["signal"]


def run_strategy(df: pd.DataFrame, symbol: str, strategy: str = "trend") -> Setup:
    """Evaluate a strategy on the LATEST bar and return a live Setup."""
    spec = STRATEGIES.get(strategy)
    if spec is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(STRATEGIES)}")

    df = add_all(df)
    last = df.iloc[-1]
    triggered = bool(spec["signal"](df).iloc[-1])

    # Most reason-builders just need the latest row; breakout needs the whole df
    # (to look back at the prior 20-day high).
    reasons = spec["reasons"](df) if spec.get("needs_df") else spec["reasons"](last)

    setup = Setup(
        symbol=symbol.upper(),
        strategy=strategy,
        direction="long",
        triggered=triggered,
        price=round(float(last["Close"]), 2),
        reasons=reasons,
        atr=round(float(last["ATR14"]), 2),
    )

    if triggered:
        entry = round(float(last["Close"]), 2)
        stop, target = build_long_trade(entry, float(last["ATR14"]))
        setup.entry = entry
        setup.stop = stop
        setup.target = target
        setup.risk_reward = risk_reward(entry, stop, target)

    return setup
