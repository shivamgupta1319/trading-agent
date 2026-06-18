"""
data.py — The DATA LAYER. Every other module gets its price data from here.

One job: hand back clean daily OHLCV (Open/High/Low/Close/Volume) candles for an
Indian NSE stock, as a pandas DataFrame. Everything that touches the network or
worries about messy data lives here, so the rest of the code can stay clean.

Why a dedicated layer?
  - One place to normalize symbols ("RELIANCE" -> "RELIANCE.NS").
  - One place to cache, so scanning 50 stocks later doesn't hammer Yahoo.
  - One place to clean bad/missing rows, so indicators never choke on NaNs.
"""

import logging
import time

import pandas as pd
import yfinance as yf

# yfinance is chatty on stderr when a symbol has no data (e.g. a delisted ticker).
# We handle those cases ourselves (skip the symbol), so quiet its logging.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# A tiny in-process cache: {(symbol, period, interval): (timestamp, DataFrame)}.
# It lives only while the program runs. Streamlit will add its own caching later;
# this just stops repeated calls in ONE run from re-hitting the network.
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 900  # 15 minutes — fine for end-of-day swing data


# ---------------------------------------------------------------------------
# Timeframes — the trading "style" maps to a candle size + how much history.
# ---------------------------------------------------------------------------
# The SAME indicators and strategies work on any candle size: an EMA50 is "50 bars"
# whether those bars are 15 minutes (≈ half a day) or 1 week (≈ a year). So switching
# timeframe is mostly just choosing the candle (`interval`) and history (`period`).
#
# Free Yahoo data limits how far back intraday goes (only ~60 days of 15m), which is
# why each timeframe has its own sensible period. `min_bars` guards against running
# indicators/backtests on too little data.
TIMEFRAMES = {
    "intraday":   {"interval": "15m", "period": "30d", "min_bars": 80,
                   "label": "Intraday (15-min candles, day trades)"},
    "swing":      {"interval": "1d",  "period": "1y",  "min_bars": 60,
                   "label": "Swing (daily candles, days–weeks)"},
    "positional": {"interval": "1wk", "period": "5y",  "min_bars": 60,
                   "label": "Positional (weekly candles, months+)"},
}


def resolve_timeframe(timeframe: str = "swing", period: str = None):
    """Turn a timeframe name into (interval, period, min_bars).

    `period` optionally overrides the timeframe's default history window.
    """
    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Available: {list(TIMEFRAMES)}")
    return tf["interval"], (period or tf["period"]), tf["min_bars"]


def normalize_symbol(symbol: str) -> str:
    """Make sure a symbol is in Yahoo's NSE form, e.g. 'reliance' -> 'RELIANCE.NS'.

    NSE stocks on Yahoo end in '.NS' (BSE would be '.BO'). We let the user type a
    bare ticker and add the suffix for them. If they already passed a suffix
    (.NS/.BO) or a known index like '^NSEI', we leave it alone.
    """
    s = symbol.strip().upper()
    # Leave alone anything that's already a complete symbol:
    #   - already suffixed (.NS / .BO) or an index (^NSEI)
    #   - contains a '.' (some other exchange suffix) or '-' (crypto like BTC-USD).
    # NSE tickers are plain letters/digits, so only those get '.NS' appended.
    if s.endswith(".NS") or s.endswith(".BO") or s.startswith("^") or "." in s or "-" in s:
        return s
    return f"{s}.NS"


def get_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch clean OHLCV candles for one NSE symbol.

    period:   how far back — "6mo", "1y", "2y", "5y", etc.
    interval: candle size — "1d" (daily) for swing trading. (Intraday like "15m"
              comes in a later phase; the parameter is here so we're ready.)

    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.
    Raises ValueError with a clear message if no usable data comes back.
    """
    sym = normalize_symbol(symbol)
    cache_key = (sym, period, interval)

    # Serve from cache if we fetched this recently.
    cached = _CACHE.get(cache_key)
    if cached is not None:
        fetched_at, df = cached
        if time.monotonic() - fetched_at < _CACHE_TTL_SECONDS:
            return df

    # auto_adjust=True gives split/dividend-adjusted prices — what you want for
    # consistent indicator math across corporate actions.
    raw = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=True)

    if raw is None or raw.empty:
        raise ValueError(
            f"No data for '{sym}'. Check the symbol (NSE tickers look like 'TCS.NS')."
        )

    # --- hygiene -----------------------------------------------------------
    # Keep only the columns we care about, drop any rows missing a price, and
    # make the index timezone-naive so date math elsewhere stays simple.
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if df.empty:
        raise ValueError(f"Data for '{sym}' was all empty/invalid after cleaning.")

    _CACHE[cache_key] = (time.monotonic(), df)
    return df


def get_many_ohlcv(symbols, period: str = "1y", interval: str = "1d") -> dict:
    """Fetch OHLCV for MANY symbols in a single batched network call.

    Scanning 50 stocks one-by-one means 50 round-trips to Yahoo — slow and prone to
    rate-limiting. `yf.download` fetches them all at once, which is far faster and
    gentler. Returns {symbol: cleaned DataFrame}; symbols that fail are simply omitted
    (the scanner just skips them).
    """
    syms = [normalize_symbol(s) for s in symbols]

    # group_by="ticker" gives a column layout of (symbol, field); threads parallelizes.
    raw = yf.download(
        syms, period=period, interval=interval,
        auto_adjust=True, group_by="ticker", threads=True, progress=False,
    )

    out = {}
    for sym in syms:
        try:
            # With multiple tickers the columns are a MultiIndex keyed by symbol.
            sub = raw[sym] if sym in raw.columns.get_level_values(0) else None
            if sub is None:
                continue
            df = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if not df.empty:
                out[sym] = df
        except Exception:
            # One bad symbol shouldn't sink the whole scan.
            continue

    return out
