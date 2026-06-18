"""
indicators.py — The ANALYTICAL CORE. Where raw prices become trading signals.

Every function here is PURE: give it a price DataFrame, it returns the same
DataFrame with new indicator columns added. No network, no surprises.

We compute the math BY HAND (just pandas .ewm/.rolling) instead of using a library.
Two reasons:
  1. The popular `pandas-ta` library is broken on modern pandas/numpy.
  2. You actually learn what these indicators ARE by reading the ~10 lines each.

Each indicator below has a one-paragraph "what & why" so you understand the signal,
not just the code.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average
# ---------------------------------------------------------------------------
# A moving average smooths price into a trend line. "Exponential" means recent
# days are weighted more than old ones, so it reacts faster than a simple average.
# Traders watch the 20 vs 50 EMA: price above both = uptrend; 20 crossing above 50
# ("golden cross") is a classic bullish signal.
def add_emas(df: pd.DataFrame, spans=(20, 50)) -> pd.DataFrame:
    for span in spans:
        df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# RSI — Relative Strength Index (Wilder, 14-period)
# ---------------------------------------------------------------------------
# Measures momentum on a 0–100 scale: how strong recent gains are vs recent losses.
# Rule of thumb: >70 = "overbought" (maybe stretched), <30 = "oversold" (maybe
# bouncing), ~50 = neutral. We use Wilder's smoothing (an EMA with alpha = 1/period),
# which is the standard every charting site uses — so our numbers match TradingView.
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()                 # day-over-day price change
    gain = delta.clip(lower=0)                  # keep gains, zero out losses
    loss = -delta.clip(upper=0)                 # keep losses (as positives)

    # Wilder's smoothing == ewm with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df[f"RSI{period}"] = 100 - (100 / (1 + rs))
    return df


# ---------------------------------------------------------------------------
# MACD — Moving Average Convergence Divergence
# ---------------------------------------------------------------------------
# A momentum/trend hybrid. MACD line = fast EMA(12) − slow EMA(26). When it's
# positive and rising, short-term momentum is beating long-term = bullish. The
# "signal" line is an EMA(9) of the MACD; MACD crossing ABOVE its signal is a
# common buy trigger. The histogram (MACD − signal) shows that gap visually.
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


# ---------------------------------------------------------------------------
# ATR — Average True Range (14-period)
# ---------------------------------------------------------------------------
# A pure VOLATILITY measure: on average, how many rupees does this stock move in a
# day? "True Range" is the largest of (high−low), |high−prevClose|, |low−prevClose|,
# so it accounts for gaps. We don't trade ATR directly — we use it later to size
# stop-losses (a wider-moving stock needs a wider stop). Wilder-smoothed again.
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df[f"ATR{period}"] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------
# A volatility envelope: a 20-day average (middle band) with an upper/lower band
# 2 standard deviations away. Price tends to stay inside; tagging the LOWER band in
# an uptrend is a classic "buy the dip" (mean-reversion) signal, while riding the
# UPPER band signals strong momentum. Bands widen in volatile times, pinch in calm ones.
def add_bbands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + num_std * std
    df["BB_lower"] = mid - num_std * std
    return df


# ---------------------------------------------------------------------------
# ADX — Average Directional Index (trend STRENGTH, 14-period)
# ---------------------------------------------------------------------------
# ADX answers "is there a trend at all?" on a 0–100 scale — NOT the direction.
# <20 = choppy/rangebound (trend strategies struggle); >25 = a real trend worth
# riding. We pair it with +DI/−DI which DO show direction. Breakout trades love a
# rising ADX. This is the most involved indicator here — built from directional movement.
def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    # Wilder smoothing (alpha = 1/period) for TR and the directional movements.
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df[f"ADX{period}"] = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    return df


# ---------------------------------------------------------------------------
# Supertrend — a trailing trend line (very popular with Indian traders)
# ---------------------------------------------------------------------------
# Plots a line that flips below price in an uptrend (acts as a trailing stop) and
# above price in a downtrend. When price closes across the line, the trend "flips."
# Built from ATR bands. Needs a small loop because each bar's band depends on the
# previous bar's — the one place pure vectorization doesn't fit cleanly.
def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]
    hl2 = (high + low) / 2

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    upper = (hl2 + multiplier * atr).to_numpy()
    lower = (hl2 - multiplier * atr).to_numpy()
    close_arr = close.to_numpy()
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n)  # 1 = uptrend, -1 = downtrend

    for i in range(1, n):
        # Carry the band forward unless price/volatility justifies tightening it.
        final_upper[i] = (
            upper[i] if (upper[i] < final_upper[i - 1] or close_arr[i - 1] > final_upper[i - 1] or np.isnan(final_upper[i - 1]))
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i] if (lower[i] > final_lower[i - 1] or close_arr[i - 1] < final_lower[i - 1] or np.isnan(final_lower[i - 1]))
            else final_lower[i - 1]
        )
        # Flip direction when price closes through the opposite band.
        if close_arr[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close_arr[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    df["ST_dir"] = direction  # +1 bullish / -1 bearish
    df["ST_line"] = np.where(direction == 1, final_lower, final_upper)
    return df


# ---------------------------------------------------------------------------
# VWAP (rolling) + relative volume
# ---------------------------------------------------------------------------
# VWAP = the average price weighted by volume. It answers "what's the fair price most
# trades happened at?" and intraday traders treat it as a magnet/support line.
#   - INTRADAY: true VWAP resets every session (each trading day starts fresh). We
#     detect intraday data (multiple bars share a calendar date) and anchor per day.
#   - DAILY/WEEKLY: a single-session VWAP makes no sense, so we use a 20-bar rolling
#     volume-weighted price as a "fair value" reference instead.
# Either way the column is named VWAP20 for consistency downstream.
def add_vwap(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3

    # If several bars share the same date, this is intraday → anchor VWAP per session.
    is_intraday = df.index.normalize().duplicated().any()
    if is_intraday:
        day = df.index.normalize()
        cum_pv = (typical * df["Volume"]).groupby(day).cumsum()
        cum_vol = df["Volume"].groupby(day).cumsum()
        df["VWAP20"] = cum_pv / cum_vol
    else:
        pv = (typical * df["Volume"]).rolling(period).sum()
        vol = df["Volume"].rolling(period).sum()
        df["VWAP20"] = pv / vol
    return df


def add_volume_stats(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    avg_vol = df["Volume"].rolling(period).mean()
    df["RelVolume"] = df["Volume"] / avg_vol  # 1.0 = average, 2.0 = double normal
    return df


# ---------------------------------------------------------------------------
# Convenience: add every indicator at once.
# Kept indicator-by-indicator above so any single one can be toggled/debugged.
# ---------------------------------------------------------------------------
def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    add_emas(df)
    add_rsi(df)
    add_macd(df)
    add_atr(df)
    add_bbands(df)
    add_adx(df)
    add_supertrend(df)
    add_vwap(df)
    add_volume_stats(df)
    return df


def latest_indicator_snapshot(df: pd.DataFrame) -> dict:
    """Return the MOST RECENT value of each indicator as a clean dict.

    This is the JSON-friendly summary the agent's tool will hand to the model:
    plain numbers plus a couple of human-readable interpretations.
    """
    df = add_all(df)
    last = df.iloc[-1]  # the latest day's row

    close = float(last["Close"])
    rsi = float(last["RSI14"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    macd = float(last["MACD"])
    macd_signal = float(last["MACD_signal"])
    atr = float(last["ATR14"])

    # Translate raw numbers into the kind of read a trader would say out loud.
    if rsi >= 70:
        rsi_note = "overbought (momentum stretched, watch for a pullback)"
    elif rsi <= 30:
        rsi_note = "oversold (beaten down, watch for a bounce)"
    else:
        rsi_note = "neutral"

    if close > ema20 > ema50:
        trend_note = "uptrend (price above rising 20 & 50 EMA)"
    elif close < ema20 < ema50:
        trend_note = "downtrend (price below falling 20 & 50 EMA)"
    else:
        trend_note = "sideways / transitioning"

    macd_note = "bullish (MACD above signal)" if macd > macd_signal else "bearish (MACD below signal)"

    # New (Phase 4) indicators — guarded with a helper since early bars can be NaN.
    def num(col):
        v = last[col]
        return None if pd.isna(v) else round(float(v), 2)

    adx = num("ADX14")
    if adx is None:
        adx_note = "n/a"
    elif adx >= 25:
        adx_note = "strong trend (ADX >= 25)"
    elif adx < 20:
        adx_note = "weak/choppy (ADX < 20, trend trades risky)"
    else:
        adx_note = "developing trend"

    st_dir = last["ST_dir"]
    supertrend_note = "bullish (price above Supertrend)" if st_dir == 1 else "bearish (price below Supertrend)"

    rel_vol = num("RelVolume")
    if rel_vol is None:
        volume_note = "n/a"
    elif rel_vol >= 1.5:
        volume_note = f"heavy volume ({rel_vol}x average)"
    elif rel_vol < 0.7:
        volume_note = f"light volume ({rel_vol}x average)"
    else:
        volume_note = f"normal volume ({rel_vol}x average)"

    bb_lower, bb_upper = num("BB_lower"), num("BB_upper")
    if bb_lower is not None and close <= bb_lower:
        bb_note = "at/below lower band (stretched down — possible bounce)"
    elif bb_upper is not None and close >= bb_upper:
        bb_note = "at/above upper band (strong momentum or overextended)"
    else:
        bb_note = "inside the bands (normal range)"

    return {
        "close": round(close, 2),
        "rsi14": round(rsi, 1),
        "rsi_note": rsi_note,
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "trend_note": trend_note,
        "macd": round(macd, 2),
        "macd_signal": round(macd_signal, 2),
        "macd_note": macd_note,
        "atr14": round(atr, 2),
        "atr_pct_of_price": round(atr / close * 100, 2),  # daily volatility as % of price
        "adx14": adx,
        "adx_note": adx_note,
        "supertrend_note": supertrend_note,
        "rel_volume": rel_vol,
        "volume_note": volume_note,
        "bb_note": bb_note,
    }
