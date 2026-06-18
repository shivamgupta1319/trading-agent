# 📈 NSE Trading Agent

A **learn-by-building AI agent** that analyzes Indian (NSE) stocks. It computes real
technical indicators, applies rule-based trading strategies, produces concrete trade
setups (entry / stop-loss / target / risk:reward / position size), **backtests** every
strategy for an honest win-rate, scans the whole Nifty 50 for the best opportunities, and
wraps it all in a Streamlit dashboard with an AI chat that drives the same tools.

It was built step-by-step as a way to _learn how AI agents actually work_ — so the code is
heavily commented and each piece does one clear job.

> ## ⚠️ Important
>
> This is **educational decision-support, NOT financial advice**, and it is **read-only** —
> it can never place a trade. An LLM cannot predict markets. Every suggestion here is
> rule-based and backtested, the edges are modest, and **past performance does not
> guarantee future results**. Do not trade real money based on this tool.

---

## Table of contents

- [What it does](#what-it-does)
- [The core idea: what _is_ an agent?](#the-core-idea-what-is-an-agent)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Project architecture](#project-architecture)
- [The 7 tools](#the-7-tools-the-agent-can-call)
- [Strategies](#strategies-long-only)
- [Timeframes](#timeframes)
- [Risk management math](#risk-management-math)
- [How backtesting works](#how-backtesting-works)
- [The scanner](#the-scanner)
- [Using the dashboard](#using-the-dashboard)
- [Using the CLI](#using-the-cli)
- [Honest limitations](#honest-limitations)
- [How it was built (the learning journey)](#how-it-was-built-the-learning-journey)
- [Extending it](#extending-it)
- [Troubleshooting](#troubleshooting)
- [Tech stack](#tech-stack)

---

## What it does

- **Live technical analysis** — RSI, EMA20/50, MACD, ATR, ADX, Supertrend, Bollinger
  Bands, VWAP and relative volume, with plain-English interpretation.
- **Concrete trade setups** — not just "looks bullish," but entry, stop-loss, target, a
  computed risk:reward ratio, and how many shares to buy for a given risk %.
- **Backtesting** — replays each strategy over history and reports win-rate and
  expectancy, so suggestions carry evidence, not vibes.
- **Whole-market scanner** — ranks the Nifty 50 by which stocks have a triggered setup
  _and_ a real historical edge today.
- **Three timeframes** — intraday (15-min), swing (daily), positional (weekly).
- **An AI agent layer** — ask questions in plain English; the model decides which tools to
  call, remembers the conversation, and streams its thinking.

---

## The core idea: what _is_ an agent?

Strip away the hype and an AI agent is **a loop around a model that can call tools**:

```
            ┌─────────────────────────────────────────────┐
            │  send conversation + list of tools to model  │
            └───────────────────────┬─────────────────────┘
                                    │
                       model replies with…
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                        ▼
          tool call(s)                            final text answer
                │                                        │
       YOUR code runs the tool                        DONE ✓
       and appends the result
                │
                └──────────── loop back up ─────────────┘
```

A chatbot answers in one shot. An **agent** runs this loop, so the model can gather data
across several steps and decide its own next move. That loop lives in
[`agent.py`](agent.py) → `run_conversation()`. Everything else in this project is just
**tools the agent can call** (in [`tools.py`](tools.py)) and the **functions behind them**.

The pattern for adding any new capability is always the same three steps:

1. Write a plain Python function.
2. Describe it with a JSON schema (so the model knows it exists).
3. Register it in the dispatch table.

We did this 7 times.

---

## Quick start

### Prerequisites

- Python 3.10+ (developed on 3.12)
- An [OpenRouter](https://openrouter.ai) API key (it speaks the OpenAI API and gives access
  to many models — Gemini, Llama, GPT, etc. — through one endpoint)

### Install

```bash
cd trading-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env and paste your OpenRouter key
```

Market data comes from `yfinance` (free, **no key needed**). NSE symbols can be typed bare
(`RELIANCE`, `TCS`) — the `.NS` suffix is added automatically.

### Run the dashboard (recommended)

```bash
streamlit run app.py
```

### Or run from the command line

```bash
python main.py "Scan the Nifty 50 for the best trend setups today."
python main.py "I have 2 lakh. Is there a swing trade in BAJFINANCE? Give entry, stop, target, shares."
python main.py "Backtest the breakout strategy on RELIANCE."
```

---

## Configuration

Everything configurable lives in `.env` (copied from `.env.example`):

| Variable             | Purpose                                 | Default                   |
| -------------------- | --------------------------------------- | ------------------------- |
| `OPENROUTER_API_KEY` | Your OpenRouter key (required)          | —                         |
| `MODEL`              | Any tool-capable model id on OpenRouter | `google/gemini-2.5-flash` |

**Choosing a model.** Any model that supports tool/function-calling works. Good options:

- `google/gemini-2.5-flash` — cheap, fast, reliable (default)
- `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet` — also solid
- `meta-llama/llama-3.3-70b-instruct:free`, `qwen/qwen3-coder:free` — free, but often
  rate-limited upstream

> 💡 If you hit a `402` error ("requires more credits"), your balance is low — the app caps
> responses at `max_tokens=1024` to keep cost tiny, but a near-zero balance can still fail.

**Theme.** The dark look lives in [`.streamlit/config.toml`](.streamlit/config.toml) —
change the colors there to taste.

---

## Project architecture

One job per file, no circular dependencies:

```
trading-agent/
├── data.py          Fetch + cache NSE OHLCV; symbol normalization; batch download; timeframes
├── universe.py      The Nifty 50 list (hardcoded, dated)
├── indicators.py    Manual indicator math (RSI, EMA, MACD, ATR, Bollinger, ADX, Supertrend, VWAP, volume)
├── risk.py          Stop-loss, target, risk:reward, position sizing
├── strategies.py    Rule-based strategies → a Setup (trend / breakout / mean_reversion)
├── backtest.py      Walk-forward backtester (anti-look-ahead) → win-rate & expectancy
├── scanner.py       Run a strategy across the Nifty 50, rank by backtested edge
├── tools.py         Wraps all of the above as the 7 tools the agent can call
├── agent.py         The tool-calling loop + system prompt
├── main.py          CLI entry point
├── app.py           Streamlit dashboard: Chart · Scanner · Chat
├── requirements.txt
├── .env.example     Copy to .env and add your key
└── .streamlit/
    └── config.toml  Dark theme
```

**Dependency flow** (arrows point to what a file depends on):

```
universe ─┐
data ─────┼─→ indicators ─→ strategies ─→ backtest ─→ scanner
          │                      ▲            │           │
          └──────── risk ────────┘            │           │
                                              ▼           ▼
                            tools.py ←─────────────────────
                                │
                            agent.py ←── main.py
                                │
                            app.py
```

### Module deep-dive

- **`data.py`** — the only file that touches the network. `get_ohlcv()` fetches and cleans
  candles for one symbol (drops NaNs, strips timezone, caches for 15 min). `get_many_ohlcv()`
  batch-downloads many symbols in a single call (used by the scanner to avoid rate limits).
  `normalize_symbol()` adds `.NS` to bare tickers (and leaves crypto/`.`/`-` symbols alone).
  `resolve_timeframe()` maps a timeframe name to `(interval, period, min_bars)`.
- **`indicators.py`** — every indicator is computed **by hand** with pandas (no TA library).
  This is deliberate: the popular `pandas-ta` breaks on modern pandas/numpy, and writing the
  ~10 lines per indicator is how you actually learn what they are. `add_all(df)` adds every
  indicator column; `latest_indicator_snapshot(df)` returns the latest values + notes.
- **`risk.py`** — pure arithmetic for the numbers that make trading survivable: ATR-based and
  swing-low stops, target from a target R:R, the realized R:R, and position sizing.
- **`strategies.py`** — each strategy is a **vectorized signal function** (`df → boolean
Series`, true on every bar where it triggers) plus a reason-builder. Crucially, the _same_
  signal function feeds both the live setup and the backtest, so they can never drift apart.
- **`backtest.py`** — replays signals forward through history (see [below](#how-backtesting-works)).
- **`scanner.py`** — runs a strategy across the universe and ranks the survivors by edge.
- **`tools.py`** — thin wrappers that turn the above into JSON-friendly dicts, plus the JSON
  schemas the model reads. This is the agent's whole "API."
- **`agent.py`** — `new_conversation()` starts a chat (system prompt only); `run_conversation()`
  runs the tool-calling loop over a growing message list (this is the memory); `run_agent()`
  is a single-turn convenience wrapper for the CLI.

---

## The 7 tools the agent can call

| Tool                | Arguments                                                  | What it returns                                                                  |
| ------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `get_quote`         | `symbol`                                                   | Latest price, day change, year high/low, currency                                |
| `get_price_history` | `symbol`, `period`                                         | Summarized recent history (start/end/high/low/% change)                          |
| `get_indicators`    | `symbol`, `timeframe`                                      | All indicators + plain-English trend/momentum/volatility notes                   |
| `analyze_setup`     | `symbol`, `strategy`, `timeframe`, `capital?`, `risk_pct?` | A full setup: triggered?, reasons, entry/stop/target/R:R, optional position size |
| `position_size`     | `entry`, `stop`, `capital`, `risk_pct?`                    | Shares to buy so a stop-out loses only `risk_pct`% of capital                    |
| `backtest_strategy` | `symbol`, `strategy`, `timeframe`                          | Trades, win-rate, avg win/loss (in R), expectancy + a plain verdict              |
| `scan_market`       | `strategy`, `top_n?`                                       | Ranked Nifty 50 setups triggering today, each with its backtested stats          |

The model picks which to call — you never hard-code "if the user asks X, call Y."

---

## Strategies (long-only)

All three are long-only (buy expecting a rise). Each produces a `Setup` whether or not it
triggers, with the exact pass/fail reasons.

| Strategy           | Philosophy                                            | Rules (all must be true)                                             |
| ------------------ | ----------------------------------------------------- | -------------------------------------------------------------------- |
| **trend**          | "the trend is your friend" — buy established uptrends | `close > EMA20 > EMA50` · `RSI` between 50–70 · `MACD > signal`      |
| **breakout**       | "buy strength" — buy new highs with conviction        | `close >` 20-day high · relative volume `> 1.2` · `ADX > 20`         |
| **mean_reversion** | "buy the dip in an uptrend"                           | `close > EMA50` · `RSI < 45` · `close <` 20-day mean (Bollinger mid) |

When a strategy triggers, the trade is shaped the same way for all of them (in
`strategies.py`):

- **Entry** = latest close
- **Stop** = entry − (2 × ATR) — a volatility-adjusted "I was wrong" level
- **Target** = a 2:1 reward-to-risk distance above entry

Adding a 4th strategy is one signal function + one reason-builder + one line in the
`STRATEGIES` registry — and it's instantly backtestable and scannable.

---

## Timeframes

The same indicators and strategies run on three timeframes, because an "EMA50" is just _50
bars_ — 50 fifteen-minute candles or 50 weeks, the math doesn't care.

| Timeframe         | Candle | History pulled | Min bars | Use for             |
| ----------------- | ------ | -------------- | -------- | ------------------- |
| `intraday`        | 15-min | ~30 days       | 80       | same-day trades     |
| `swing` (default) | daily  | ~1 year        | 60       | days-to-weeks holds |
| `positional`      | weekly | ~5 years       | 60       | months+ holds       |

`get_indicators`, `analyze_setup`, and `backtest_strategy` accept a `timeframe`, and the
agent infers it from your wording ("day trade today" → intraday, "for the long term" →
positional, otherwise swing). VWAP is **session-anchored** (resets each day) on intraday
and a rolling reference on daily/weekly.

> The **scanner is daily/swing only** — scanning all 50 stocks on intraday data would be
> slow and noisy. A deliberate choice.

---

## Risk management math

In [`risk.py`](risk.py). This is the part that separates trading from gambling.

```
ATR stop (long):    stop   = entry − atr_mult × ATR          (atr_mult = 2.0)
Target:             target = entry + rr_target × (entry − stop)   (rr_target = 2.0)
Realized R:R:       rr     = (target − entry) / (entry − stop)

Position sizing (risk a fixed % of capital):
    risk_amount    = capital × risk_pct / 100        e.g. 1% of ₹1,00,000 = ₹1,000
    per_share_risk = entry − stop
    shares         = floor(risk_amount / per_share_risk)   (rounded down — never over-risk)
```

If the share count would cost more than your capital, it's capped to what the cash allows
(`capital_capped: true`). The golden rule baked in: **never risk more than ~1–2% of your
account on a single trade**, so a losing streak is survivable.

---

## How backtesting works

In [`backtest.py`](backtest.py). A strategy that "looks smart" is worthless until it's been
tested on history. The backtester replays each strategy bar-by-bar and measures the truth.

**The #1 trap — look-ahead bias.** It's terrifyingly easy to accidentally use information
you wouldn't have had in real time, which makes a useless strategy look brilliant. We avoid
it with one hard rule:

> The entry signal is computed on bar _t_ (today's close), but we **enter at bar _t+1_'s
> open** (tomorrow's open) — you can never act on a candle before it closes.

**How a trade is scored.** From the entry, we walk forward until the **stop** or **target**
is hit. Results are measured in **R-multiples** (R = one unit of risk = entry − stop):

- target hit first → **+2R** (because targets are set at 2:1)
- stop hit first → **−1R**
- if both are touched on the same day, we pessimistically assume the **stop** hit first

We trade one position at a time (no overlapping trades). The headline number is
**expectancy** = average R per trade. Positive expectancy = a historical edge.

**The control experiment.** With a 2:1 target, blind/random buying should win only ~33% of
the time and have ~zero expectancy. We verified an "always-buy" control does exactly that —
which proves the backtester isn't secretly cheating. Real strategies should beat it.

> ⚠️ A single stock often produces only a handful of trades — too small a sample to trust.
> Lean on the **aggregate across many stocks and years**, not one number.

---

## The scanner

In [`scanner.py`](scanner.py). For a chosen strategy it:

1. Batch-downloads the whole Nifty 50 in one call.
2. Runs the live strategy on each stock — keeping only those triggering **today**.
3. Backtests each survivor so we know that signal's historical quality.
4. **Ranks** them by backtested expectancy (best edge first).

So a scan result isn't just "this triggered" — it's "this triggered today **and** the
strategy has a real edge on this stock." Rows backed by fewer than 8 backtest trades are
flagged `low (few backtest trades)`. Any symbols Yahoo can't return are reported in
`missing_symbols` rather than silently dropped.

---

## Using the dashboard

`streamlit run app.py` opens three tabs (controls in the sidebar: stock, timeframe,
strategy):

- **📈 Chart** — current metrics (price + % change, RSI, ADX, ATR%), the live setup verdict
  for the chosen strategy (with entry/stop/target/R:R when triggered), an expandable
  rules breakdown, and an interactive Plotly chart: candles + EMA20/50 + Supertrend, a
  colored volume sub-chart, RSI, and MACD. Weekend/overnight gaps are removed.
- **🔎 Scanner** — click **Run scan** to rank the Nifty 50 for the selected strategy, shown
  as a formatted table (₹ prices, a win-rate bar, signed expectancy).
- **💬 Ask the Agent** — chat with the agent. It **remembers** the conversation (so "yes, go
  ahead" or "now backtest that one" works) and **streams its tool calls live** as it works.
  The input is pinned to the bottom; **🧹 New chat** resets it.

---

## Using the CLI

`main.py` answers a single question and prints the agent's tool calls as it goes:

```bash
python main.py "How is RELIANCE doing on the daily timeframe?"
python main.py "Find me a breakout setup in Nifty 50 and prove it with a backtest."
python main.py "I have 1 lakh, 1.5% risk. How many TCS shares if I enter 2200, stop 2120?"
python main.py "Compare trend vs mean_reversion on LT over the long term."
```

---

## Honest limitations

- **Not financial advice, not a predictor.** This is rule-based analysis with backtested
  context — nothing more.
- **Free data caveats.** Yahoo data can be ~15 min delayed and is end-of-day for daily —
  fine for swing/positional, less so for true intraday. A few symbols (e.g. `TATAMOTORS.NS`,
  `LTIM.NS`) intermittently return no data and are skipped.
- **Small samples are noisy.** Per-stock backtests can have very few trades. Treat single
  numbers with suspicion.
- **The Nifty 50 list is hardcoded** (it reconstitutes ~twice a year) — update `universe.py`
  by hand when needed.
- **Long-only.** No short-selling yet (the structure leaves room for it).

---

## How it was built (the learning journey)

The project grew in small, independently-runnable phases — a good template for building any
agent:

1. **Live indicators** — a data layer + manual indicators + the first tool.
2. **Risk + a strategy** — turn indicators into a concrete, risk-defined setup.
3. **Backtesting** — the credibility layer; every setup gains an honest track record.
4. **More strategies & indicators** — breakout, mean-reversion, and the "pro" indicators.
5. **The scanner** — survey the whole Nifty 50, ranked by edge.
6. **The Streamlit GUI** — charts, scanner table, and a chat over all the tools.

Then: multi-turn chat memory, live streaming of the agent's thinking, multi-timeframe
support, and UI polish.

---

## Extending it

**Add an indicator** — write a `add_x(df)` function in `indicators.py`, call it inside
`add_all()`, optionally surface it in `latest_indicator_snapshot()`.

**Add a strategy** — in `strategies.py`, write a vectorized `x_signal(df) → bool Series` and
a reason-builder, then add one line to the `STRATEGIES` registry. It's immediately usable by
`analyze_setup`, `backtest_strategy`, and `scan_market`.

**Add a tool** — write the function in `tools.py`, add its JSON schema to `TOOL_SCHEMAS`, and
register it in `TOOL_FUNCTIONS`. The agent will start using it on its own.

**Add a timeframe** — add an entry to `TIMEFRAMES` in `data.py`.

Natural next features: a watchlist/portfolio view, price alerts, more strategies,
short-selling, or a broker API (Zerodha Kite / Upstox / Angel One) for real-time data.

---

## Troubleshooting

| Symptom                              | Cause / fix                                                      |
| ------------------------------------ | ---------------------------------------------------------------- |
| `Missing OPENROUTER_API_KEY`         | Copy `.env.example` to `.env` and paste your key                 |
| `404 No endpoints found for <model>` | The `MODEL` id is wrong — pick one from openrouter.ai/models     |
| `402 requires more credits`          | Low OpenRouter balance; add credits or switch to a `:free` model |
| `429 rate-limited`                   | Free models share an upstream limit — retry, or use a paid model |
| A stock shows "no data"              | Yahoo occasionally drops a symbol; it's skipped automatically    |
| Scan is slow the first time          | It downloads 50 stocks once, then caches for 15 minutes          |

---

## Tech stack

- **Python** + **pandas** / **numpy** — data & indicator math (computed by hand)
- **yfinance** — free NSE market data
- **openai** SDK → **OpenRouter** — the model client (works with Gemini, GPT, Llama, …)
- **Streamlit** + **Plotly** — the dashboard and interactive charts

---

_Built to learn how AI agents work — a model, some tools, and a loop. Use it to learn, not
to trade real money._
