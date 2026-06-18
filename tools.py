"""
tools.py — The "hands" of the agent.

A tool is just a normal Python function. The ONLY extra thing an agent needs is a
machine-readable *description* of each tool (the JSON schema at the bottom) so the
model knows the tool exists, what it does, and what arguments it takes.

The model never runs these functions itself. It only *asks* to run them by name,
and OUR code (in agent.py) actually executes them and hands back the result.

These tools are READ-ONLY: they fetch market data. Nothing here can place a trade
or move money. That's deliberate — a learning agent should never touch real orders.
"""

import yfinance as yf

from data import get_ohlcv, normalize_symbol, resolve_timeframe, TIMEFRAMES
from indicators import latest_indicator_snapshot
from strategies import run_strategy, STRATEGIES
from risk import position_size as compute_position_size
from backtest import backtest_strategy as run_backtest
from scanner import scan as run_scan


# ---------------------------------------------------------------------------
# Tool 1: get a current quote for a symbol (stock or crypto)
# ---------------------------------------------------------------------------
def get_quote(symbol: str) -> dict:
    """Return the latest price and basic stats for a ticker.

    Examples of symbols: "AAPL" (Apple), "TSLA" (Tesla), "BTC-USD" (Bitcoin),
    "ETH-USD" (Ethereum). Crypto uses the "-USD" suffix on Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(normalize_symbol(symbol))  # 'RELIANCE' -> 'RELIANCE.NS'
        info = ticker.fast_info  # fast, lightweight snapshot

        price = info.last_price
        prev_close = info.previous_close
        # Day change as a percent, guarding against missing/zero data.
        day_change_pct = None
        if price is not None and prev_close:
            day_change_pct = round((price - prev_close) / prev_close * 100, 2)

        return {
            "symbol": symbol,
            "price": round(price, 2) if price is not None else None,
            "previous_close": round(prev_close, 2) if prev_close else None,
            "day_change_pct": day_change_pct,
            "year_high": round(info.year_high, 2) if info.year_high else None,
            "year_low": round(info.year_low, 2) if info.year_low else None,
            "currency": info.currency,
        }
    except Exception as e:
        # Returning the error (instead of crashing) lets the MODEL see what went
        # wrong and react — e.g. tell the user the symbol looks invalid.
        return {"error": f"Could not fetch quote for '{symbol}': {e}"}


# ---------------------------------------------------------------------------
# Tool 2: get recent price history so the agent can reason about a trend
# ---------------------------------------------------------------------------
def get_price_history(symbol: str, period: str = "1mo") -> dict:
    """Return summarized recent price history for a ticker.

    `period` is a Yahoo Finance window like: "5d", "1mo", "3mo", "6mo", "1y".
    We summarize instead of dumping every data point, so the model gets a clean
    signal (start, end, high, low, overall change) instead of noise.
    """
    try:
        ticker = yf.Ticker(normalize_symbol(symbol))  # 'RELIANCE' -> 'RELIANCE.NS'
        hist = ticker.history(period=period)

        if hist.empty:
            return {"error": f"No history found for '{symbol}' over '{period}'."}

        closes = hist["Close"]
        start_price = float(closes.iloc[0])
        end_price = float(closes.iloc[-1])
        change_pct = round((end_price - start_price) / start_price * 100, 2)

        return {
            "symbol": symbol,
            "period": period,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "period_high": round(float(closes.max()), 2),
            "period_low": round(float(closes.min()), 2),
            "change_pct": change_pct,
            "data_points": len(closes),
        }
    except Exception as e:
        return {"error": f"Could not fetch history for '{symbol}': {e}"}


# ---------------------------------------------------------------------------
# Tool 3: technical indicators for swing analysis (Indian NSE stocks)
# ---------------------------------------------------------------------------
def get_indicators(symbol: str, timeframe: str = "swing") -> dict:
    """Compute the latest technical indicators for an NSE stock.

    Returns RSI, EMA20/50, MACD, ATR, ADX, Supertrend, Bollinger and volume notes —
    the raw material for a trade view. The indicators adapt to the chosen timeframe.

    `symbol` can be bare ('RELIANCE') or suffixed ('RELIANCE.NS'); we normalize it.
    `timeframe` is 'intraday' (15-min), 'swing' (daily), or 'positional' (weekly).
    """
    try:
        interval, period, min_bars = resolve_timeframe(timeframe)
        df = get_ohlcv(symbol, period=period, interval=interval)
        if len(df) < min_bars:
            return {"error": f"Not enough {timeframe} history for '{symbol}' to compute indicators."}
        snapshot = latest_indicator_snapshot(df)
        snapshot["symbol"] = symbol.upper()
        snapshot["timeframe"] = timeframe
        return snapshot
    except Exception as e:
        return {"error": f"Could not compute indicators for '{symbol}': {e}"}


# ---------------------------------------------------------------------------
# Tool 4: analyze a full trade SETUP (entry / stop / target / risk:reward [+ size])
# ---------------------------------------------------------------------------
def analyze_setup(
    symbol: str,
    strategy: str = "trend",
    timeframe: str = "swing",
    capital: float = None,
    risk_pct: float = 1.0,
) -> dict:
    """Run a trading strategy on a stock and return a concrete trade setup.

    Returns whether the strategy is triggered right now, the reasons (each rule
    passed/failed), and — if triggered — entry, stop-loss, target and risk:reward.
    If `capital` is given, also returns a position size that risks only `risk_pct`%
    of that capital. `timeframe` is 'intraday', 'swing' (default), or 'positional'.
    """
    try:
        interval, period, min_bars = resolve_timeframe(timeframe)
        df = get_ohlcv(symbol, period=period, interval=interval)
        if len(df) < min_bars:
            return {"error": f"Not enough {timeframe} history for '{symbol}' to analyze a setup."}

        setup = run_strategy(df, symbol, strategy)
        result = setup.to_dict()
        result["timeframe"] = timeframe

        # If there's a live setup and the user told us their capital, size the trade.
        if setup.triggered and capital:
            result["position_size"] = compute_position_size(
                setup.entry, setup.stop, capital, risk_pct
            )
        return result
    except Exception as e:
        return {"error": f"Could not analyze setup for '{symbol}': {e}"}


# ---------------------------------------------------------------------------
# Tool 5: standalone position-size calculator (no data fetch needed)
# ---------------------------------------------------------------------------
def position_size(entry: float, stop: float, capital: float, risk_pct: float = 1.0) -> dict:
    """How many shares to buy so a stop-out loses only `risk_pct`% of `capital`.

    Use this when the user gives their own entry and stop and asks "how many shares
    should I buy?" for a given account size and risk tolerance.
    """
    return compute_position_size(entry, stop, capital, risk_pct)


# ---------------------------------------------------------------------------
# Tool 6: backtest a strategy on a stock's history (the evidence behind a setup)
# ---------------------------------------------------------------------------
def backtest_strategy(symbol: str, strategy: str = "trend", timeframe: str = "swing") -> dict:
    """Replay a strategy over history and return its honest track record.

    Returns number of trades, win-rate, average winner/loser (in R = risk units),
    and expectancy (average R per trade — positive means a historical edge).
    Use this whenever the user asks "does this work / is it reliable / what's the
    win-rate", or to back up any setup you present. `timeframe` is 'intraday',
    'swing' (default), or 'positional'.
    """
    try:
        interval, period, min_bars = resolve_timeframe(timeframe)
        df = get_ohlcv(symbol, period=period, interval=interval)
        if len(df) < min_bars:
            return {"error": f"Not enough {timeframe} history for '{symbol}' to backtest meaningfully."}
        stats = run_backtest(df, symbol, strategy, period=f"{timeframe}/{period}")
        result = stats.to_dict()
        # A plain-English read so the model frames it honestly.
        if result["trades"] == 0:
            result["verdict"] = "No trades triggered in this period — not enough data to judge."
        elif result["expectancy_r"] > 0:
            result["verdict"] = (
                f"Positive expectancy ({result['expectancy_r']}R per trade) over "
                f"{result['trades']} trades — a historical edge, but past results don't guarantee future ones."
            )
        else:
            result["verdict"] = (
                f"Negative/zero expectancy ({result['expectancy_r']}R per trade) over "
                f"{result['trades']} trades — no historical edge on this stock."
            )
        return result
    except Exception as e:
        return {"error": f"Could not backtest '{symbol}': {e}"}


# ---------------------------------------------------------------------------
# Tool 7: scan the whole Nifty 50 for the best setups right now
# ---------------------------------------------------------------------------
def scan_market(strategy: str = "trend", top_n: int = 10, period: str = "3y") -> dict:
    """Scan all of the Nifty 50 for stocks triggering a strategy TODAY, ranked by
    each stock's backtested edge (expectancy).

    Use this for "what should I look at today", "scan the market", "best breakout
    setups now", etc. Returns how many stocks triggered and the top-ranked ones with
    entry/stop/target plus their historical win-rate and expectancy.
    """
    try:
        return run_scan(strategy=strategy, period=period, top_n=top_n)
    except Exception as e:
        return {"error": f"Scan failed: {e}"}


# ---------------------------------------------------------------------------
# The "dispatch table": maps a tool NAME (string the model sends) -> the function.
# agent.py uses this to actually run whatever tool the model asked for.
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "get_quote": get_quote,
    "get_price_history": get_price_history,
    "get_indicators": get_indicators,
    "analyze_setup": analyze_setup,
    "position_size": position_size,
    "backtest_strategy": backtest_strategy,
    "scan_market": scan_market,
}


# ---------------------------------------------------------------------------
# The schemas: how we DESCRIBE the tools to the model.
# This is plain JSON following the OpenAI "tools" format (OpenRouter speaks it too).
# Each schema must match a function in TOOL_FUNCTIONS above by `name`.
# Good descriptions matter a LOT — they're how the model decides when to call a tool.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": (
                "Get the latest price and day's stats for a stock or crypto symbol. "
                "Use this to find out what something is trading at right now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. 'AAPL', 'TSLA', or 'BTC-USD' for Bitcoin.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Get summarized recent price history (start, end, high, low, % change) "
                "for a symbol. Use this to understand the recent trend before giving a view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. 'AAPL' or 'BTC-USD'.",
                    },
                    "period": {
                        "type": "string",
                        "description": "Time window: one of '5d', '1mo', '3mo', '6mo', '1y'.",
                        "enum": ["5d", "1mo", "3mo", "6mo", "1y"],
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicators",
            "description": (
                "Compute technical indicators (RSI, EMA20/50, MACD, ATR, ADX, Supertrend, "
                "Bollinger, volume) for an Indian NSE stock, with plain-English notes on "
                "trend, momentum and volatility. Accepts bare tickers like 'RELIANCE' or "
                "'TCS' (auto-converted to NSE)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE ticker, e.g. 'RELIANCE', 'TCS', 'INFY' (the '.NS' suffix is optional).",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "intraday (15-min, day trades), swing (daily, days–weeks), positional (weekly, months+).",
                        "enum": list(TIMEFRAMES.keys()),
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_setup",
            "description": (
                "Run a swing-trading strategy on an NSE stock and return a concrete trade "
                "setup: whether it's triggered now, the reasons each rule passed/failed, and "
                "(if triggered) entry, stop-loss, target and risk:reward. If the user gives "
                "their capital, also returns a position size. Use this for 'is there a trade "
                "in X', 'what's the setup', or 'where would I enter/exit'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE ticker, e.g. 'RELIANCE', 'TCS' (the '.NS' suffix is optional).",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Which strategy to apply.",
                        "enum": list(STRATEGIES.keys()),
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "intraday (15-min), swing (daily, default), or positional (weekly).",
                        "enum": list(TIMEFRAMES.keys()),
                    },
                    "capital": {
                        "type": "number",
                        "description": "Optional. The user's trading capital in INR, for position sizing.",
                    },
                    "risk_pct": {
                        "type": "number",
                        "description": "Optional. Percent of capital to risk on this trade (default 1.0).",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "position_size",
            "description": (
                "Calculate how many shares to buy so that being stopped out loses only a set "
                "percent of capital. Use when the user provides their own entry, stop, capital "
                "and (optionally) risk percent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {"type": "number", "description": "Planned entry price."},
                    "stop": {"type": "number", "description": "Planned stop-loss price (below entry)."},
                    "capital": {"type": "number", "description": "Total trading capital in INR."},
                    "risk_pct": {
                        "type": "number",
                        "description": "Percent of capital to risk (default 1.0).",
                    },
                },
                "required": ["entry", "stop", "capital"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_strategy",
            "description": (
                "Replay a strategy over a stock's history and return its track record: "
                "number of trades, win-rate, average winner/loser in R, and expectancy "
                "(avg R per trade; positive = historical edge). Use to answer 'does this "
                "work / is it reliable / what's the win-rate', or to back up a setup with evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE ticker, e.g. 'RELIANCE', 'TCS' (the '.NS' suffix is optional).",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Which strategy to backtest.",
                        "enum": list(STRATEGIES.keys()),
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "intraday (15-min, ~30 days), swing (daily, ~1yr, default), positional (weekly, ~5yr).",
                        "enum": list(TIMEFRAMES.keys()),
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_market",
            "description": (
                "Scan the entire Nifty 50 for stocks that trigger a strategy TODAY, ranked "
                "by each stock's backtested edge (expectancy). Use for 'scan the market', "
                "'what are the best setups right now', 'find me breakout candidates today'. "
                "Returns the count triggered and the top-ranked setups with entry/stop/target "
                "and historical win-rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Which strategy to scan for.",
                        "enum": list(STRATEGIES.keys()),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many top-ranked setups to return (default 10).",
                    },
                },
                "required": ["strategy"],
            },
        },
    },
]
