# NSE Swing-Trading Agent — a learn-by-building AI agent

A read-only AI agent that analyzes **Indian (NSE) stocks** for swing trades: it computes
real technical indicators, applies rule-based strategies, produces concrete trade setups
(entry / stop / target / risk:reward / position size), **backtests** them for an honest
win-rate, scans the whole Nifty 50 for the best opportunities, and presents it all in a
Streamlit dashboard with an AI chat.

> ⚠️ **Educational decision-support — NOT financial advice.** It is read-only (it can
> never place a trade). An LLM cannot predict markets; every suggestion is rule-based and
> backtested, and past results don't guarantee future ones.

## The core idea

An agent is **a loop around a model that can call tools**. The model decides *which*
tools to use; your code runs them and feeds results back until it has an answer. That loop
lives in [`agent.py`](agent.py) → `run_agent()`. Everything else is tools it can call.

## Quick start

```bash
cd trading-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # paste your OpenRouter key (https://openrouter.ai/keys)
```

**The dashboard (recommended):**
```bash
streamlit run app.py
```
**Or the command line:**
```bash
python main.py "Scan the Nifty 50 for the best trend setups today."
python main.py "I have 2 lakh. Is there a swing trade in BAJFINANCE? Give entry, stop, target, shares."
python main.py "Backtest the breakout strategy on RELIANCE over 5 years."
```

Market data comes from `yfinance` (free, no key). Symbols can be bare (`RELIANCE`) — the
`.NS` suffix is added automatically.

## Architecture (one job per file)

```
data.py        Fetch + cache NSE OHLCV; symbol normalization; batch download for scans
universe.py    The Nifty 50 list
indicators.py  Manual indicator math: RSI, EMA, MACD, ATR, Bollinger, ADX, Supertrend, VWAP, volume
risk.py        Stop-loss, target, risk:reward, position sizing
strategies.py  Rule-based strategies → a Setup (trend / breakout / mean_reversion)
backtest.py    Walk-forward backtester (anti-look-ahead) → win-rate & expectancy
scanner.py     Run a strategy across the Nifty 50, rank by backtested edge
tools.py       Wraps all of the above as the 7 tools the agent can call
agent.py       The tool-calling loop + system prompt
main.py        CLI entry
app.py         Streamlit dashboard: Chart · Scanner · Chat
```

Dependency flow (no cycles): `data/universe → indicators → strategies → backtest →
scanner`; `risk` feeds strategies; `tools.py` wraps everything; `agent.py → tools.py`;
`app.py → modules + agent`.

## The 7 tools the agent can call

| Tool | What it does |
|------|--------------|
| `get_quote` | Latest price & day stats |
| `get_price_history` | Summarized recent history |
| `get_indicators` | All indicators + plain-English notes |
| `analyze_setup` | A full trade setup (entry/stop/target/RR + optional size) |
| `position_size` | Shares to buy for a given risk % |
| `backtest_strategy` | A strategy's historical win-rate & expectancy |
| `scan_market` | Best Nifty 50 setups today, ranked by edge |

## Strategies (long-only)

- **trend** — buy established uptrends (close > EMA20 > EMA50, RSI 50–70, MACD confirms)
- **breakout** — buy 20-day highs on above-average volume with ADX > 20
- **mean_reversion** — buy oversold pullbacks (RSI < 45, below the 20-day mean) in an uptrend

Every strategy is automatically backtestable and scannable — that's the payoff of the
shared design.

## Timeframes

The same indicators and strategies run on three timeframes (an EMA50 is just "50 bars"):

- **swing** (default) — daily candles, holds of days–weeks
- **intraday** — 15-min candles, same-day trades (true session-anchored VWAP)
- **positional** — weekly candles, holds of months+

`get_indicators`, `analyze_setup` and `backtest_strategy` take a `timeframe`; the agent
infers it from your wording ("day trade" → intraday, "for the long term" → positional).

## Honest notes

- Free Yahoo data may be ~15 min delayed and is end-of-day — fine for swing, not intraday.
- Backtests on a single stock often have few trades (small sample = noisy). Trust the
  aggregate across many stocks / years more than any one number.
- Indicators are computed by hand (not a library) — partly for reliability, mostly so the
  math is readable and learnable.

## Where it could go next

Intraday (minute candles, session VWAP) and positional (weekly candles) are *architected
for* — `data.py` already threads `interval`/`period` through. Other ideas: more strategies,
a portfolio view, alerts, or a broker API for real-time data.
