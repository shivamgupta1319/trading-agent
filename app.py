"""
app.py — The Streamlit dashboard. The friendly face over everything we built.

Run it with:   streamlit run app.py

It has three tabs, each wired to code from earlier phases:
  📈 Chart    — candlesticks + indicator overlays + RSI/MACD (indicators.py)
  🔎 Scanner  — ranked Nifty 50 setups for a strategy        (scanner.py)
  💬 Chat     — ask the AI agent anything; it uses all 7 tools (agent.py)

Streamlit re-runs this whole script top-to-bottom on every interaction, so we wrap
the slow bits (data fetch, scans) in @st.cache_data to keep it snappy.
"""

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from agent import new_conversation, run_conversation
from data import get_ohlcv, resolve_timeframe, TIMEFRAMES
from indicators import add_all, latest_indicator_snapshot
from scanner import scan as run_scan
from strategies import STRATEGIES, run_strategy
from universe import NIFTY_50, display_name

load_dotenv()
MODEL = os.environ.get("MODEL", "google/gemini-2.5-flash")

st.set_page_config(page_title="NSE Trading Agent", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Cached data helpers — these are the slow, network-bound calls. Caching them
# means flipping tabs or tweaking a setting doesn't re-download everything.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_indicator_df(symbol: str, timeframe: str):
    interval, period, _ = resolve_timeframe(timeframe)
    return add_all(get_ohlcv(symbol, period=period, interval=interval))


@st.cache_data(ttl=900, show_spinner=False)
def load_scan(strategy: str, period: str, top_n: int):
    return run_scan(strategy=strategy, period=period, top_n=top_n)


def summarize_result(result) -> str:
    """Shorten a tool result to a one-line preview for the live trace.

    Tool results can be big (a scan returns 10+ rows). We show a compact gist so the
    trace stays scannable; the model still gets the full result.
    """
    if isinstance(result, dict):
        if "error" in result:
            return f"⚠️ {result['error']}"
        if "top" in result:  # a scan
            return f"{result.get('triggered_count', 0)} triggered of {result.get('with_data', '?')}"
        # Otherwise show a few key fields.
        keys = list(result.keys())[:4]
        return ", ".join(f"{k}={result[k]}" for k in keys) + ("…" if len(result) > 4 else "")
    return str(result)[:120]


# A small palette so the chart's colors are consistent and on-theme.
_UP, _DN = "#00c896", "#ff4b6b"      # green / red
_EMA20, _EMA50, _MACD = "#4aa3ff", "#ffb347", "#c792ea"   # blue, orange, purple


def build_chart(df, symbol: str):
    """A 4-row Plotly figure: price+overlays, volume, RSI, MACD — dark-themed."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.52, 0.12, 0.18, 0.18],
        subplot_titles=("Price · EMA20/50 · Supertrend", "Volume", "RSI (14)", "MACD"),
    )

    # Row 1: candles + moving averages + the supertrend line.
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", showlegend=False,
        increasing_line_color=_UP, decreasing_line_color=_DN), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20",
                             line=dict(color=_EMA20, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50",
                             line=dict(color=_EMA50, width=1.2)), row=1, col=1)
    # Supertrend: green where bullish, red where bearish (mask the other side with None).
    fig.add_trace(go.Scatter(x=df.index, y=df["ST_line"].where(df["ST_dir"] == 1),
                             name="Supertrend ↑", line=dict(color=_UP, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ST_line"].where(df["ST_dir"] == -1),
                             name="Supertrend ↓", line=dict(color=_DN, width=1.6)), row=1, col=1)

    # Row 2: volume, colored green/red by up or down candle.
    vol_colors = [_UP if c >= o else _DN for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=vol_colors, showlegend=False), row=2, col=1)

    # Row 3: RSI with 70/30 guide lines.
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI",
                             line=dict(color=_MACD, width=1.2), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=_DN, dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=_UP, dash="dot", width=1), row=3, col=1)

    # Row 4: MACD line, signal, histogram (green/red).
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color=_EMA20, width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                             line=dict(color=_EMA50, width=1.2)), row=4, col=1)
    hist_colors = [_UP if v >= 0 else _DN for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist",
                         marker_color=hist_colors, showlegend=False), row=4, col=1)

    # Remove dead space: hide weekends always, and (for intraday) the overnight gap.
    # Without this, candlesticks show ugly flat stretches over non-trading time.
    is_intraday = df.index.normalize().duplicated().any()
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if is_intraday:
        rangebreaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))  # NSE 9:15–15:30
    fig.update_xaxes(rangebreaks=rangebreaks)

    fig.update_layout(
        template="plotly_dark", height=720, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    return fig


# ===========================================================================
# Sidebar — global controls
# ===========================================================================
st.sidebar.title("📈 NSE Trading Agent")
st.sidebar.caption("Educational decision-support — **not** financial advice.")

symbol_choice = st.sidebar.selectbox(
    "Stock (Nifty 50)", [display_name(s) for s in NIFTY_50], index=NIFTY_50.index("RELIANCE.NS"),
)
timeframe = st.sidebar.radio(
    "Timeframe", list(TIMEFRAMES.keys()), index=1,  # default 'swing'
    format_func=lambda t: TIMEFRAMES[t]["label"],
)
strategy = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()), index=0)
st.sidebar.caption(f"Model: `{MODEL}`")

tab_chart, tab_scan, tab_chat = st.tabs(["📈 Chart", "🔎 Scanner", "💬 Ask the Agent"])


# ===========================================================================
# Tab 1 — Chart + the live setup for the selected stock
# ===========================================================================
with tab_chart:
    try:
        df = load_indicator_df(symbol_choice, timeframe)
        snap = latest_indicator_snapshot(df)
        setup = run_strategy(df, symbol_choice, strategy)

        st.markdown(f"### {symbol_choice}  ·  {strategy}")
        st.caption(f"{TIMEFRAMES[timeframe]['label']} · last bar {df.index[-1].strftime('%d %b %Y %H:%M')}")

        # % change of the latest candle, shown as a colored delta on the price metric.
        prev_close = float(df["Close"].iloc[-2])
        last_close = float(df["Close"].iloc[-1])
        bar_chg = (last_close - prev_close) / prev_close * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price (₹)", f"{snap['close']:,.2f}", f"{bar_chg:+.2f}%")
        c2.metric("RSI (14)", snap["rsi14"], help=snap["rsi_note"])
        c3.metric("ADX (14)", snap["adx14"], help=snap["adx_note"])
        c4.metric("ATR % / bar", snap["atr_pct_of_price"], help="typical move per candle")

        # The live setup verdict for the chosen strategy.
        if setup.triggered:
            st.success(f"✅ **{strategy}** setup TRIGGERED — entry near ₹{setup.entry:,.2f}")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Entry", f"₹{setup.entry:,.2f}")
            s2.metric("Stop-loss", f"₹{setup.stop:,.2f}",
                      f"{(setup.stop - setup.entry) / setup.entry * 100:.1f}%", delta_color="inverse")
            s3.metric("Target", f"₹{setup.target:,.2f}",
                      f"{(setup.target - setup.entry) / setup.entry * 100:+.1f}%")
            s4.metric("Risk : Reward", f"{setup.risk_reward} : 1")
        else:
            st.info(f"No **{strategy}** setup on {symbol_choice} right now — see the rules below.")
        with st.expander("Why? (the rules, pass / fail)"):
            for r in setup.reasons:
                st.write(r)

        st.plotly_chart(build_chart(df, symbol_choice), width="stretch")
    except Exception as e:
        st.error(f"Could not load {symbol_choice}: {e}")


# ===========================================================================
# Tab 2 — Scanner: ranked Nifty 50 setups for the chosen strategy
# ===========================================================================
with tab_scan:
    st.subheader(f"Nifty 50 scan — `{strategy}` strategy")
    st.caption("Stocks triggering this strategy today (daily/swing), ranked by backtested expectancy.")
    if st.button("🔎 Run scan", type="primary"):
        with st.spinner("Scanning all 50 stocks…"):
            result = load_scan(strategy, "3y", 15)

        st.write(
            f"**{result['triggered_count']}** of {result['with_data']} stocks triggered "
            f"(scanned {result['scanned']})."
        )
        if result["missing_symbols"]:
            st.caption(f"⚠️ No data for: {', '.join(result['missing_symbols'])}")

        if result["top"]:
            # column_config formats the raw numbers into a clean, readable table:
            # rupee prices, a win-rate progress bar, and signed expectancy.
            st.dataframe(
                result["top"], width="stretch", hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Stock"),
                    "entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "stop": st.column_config.NumberColumn("Stop", format="₹%.2f"),
                    "target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "risk_reward": st.column_config.NumberColumn("R:R", format="%.1f"),
                    "backtest_trades": st.column_config.NumberColumn("Trades"),
                    "win_rate_pct": st.column_config.ProgressColumn(
                        "Win rate", format="%.0f%%", min_value=0, max_value=100),
                    "expectancy_r": st.column_config.NumberColumn("Expectancy", format="%+.2f R"),
                    "confidence": st.column_config.TextColumn("Confidence"),
                },
            )
        else:
            st.info("No setups triggered for this strategy right now.")


# ===========================================================================
# Tab 3 — Chat with the agent (uses all 7 tools under the hood)
# ===========================================================================
with tab_chat:
    st.subheader("💬 Ask the agent")
    st.caption("It remembers the conversation, so you can say things like \"yes, go ahead\" "
               "or \"now backtest that one\". Try: \"scan for breakout setups\".")

    # Two parallel pieces of state:
    #   - `convo` is the FULL agent conversation (system + tool calls) — the memory.
    #   - `chat_display` is just the user/assistant text we show on screen.
    if "convo" not in st.session_state:
        st.session_state.convo = new_conversation()
        st.session_state.chat_display = []

    if st.button("🧹 New chat"):
        st.session_state.convo = new_conversation()
        st.session_state.chat_display = []

    # Replay the visible conversation.
    for role, text in st.session_state.chat_display:
        with st.chat_message(role):
            st.markdown(text)

    # Process a prompt that the (bottom-pinned) input queued on the previous run.
    # The input itself lives at the TOP LEVEL of the script (see end of file) so
    # Streamlit pins it to the bottom of the page — chat_input only docks there when
    # it's NOT nested inside a tab/container. We bridge the two via session_state.
    pending = st.session_state.pop("pending_prompt", None)
    if pending:
        with st.chat_message("user"):
            st.markdown(pending)
        st.session_state.chat_display.append(("user", pending))
        # Add the new user turn to the running conversation, then let the agent
        # continue it WITH full memory of everything before.
        st.session_state.convo.append({"role": "user", "content": pending})
        with st.chat_message("assistant"):
            # A live status box that streams each tool call as the agent makes it —
            # you watch the think→act→observe loop happen in real time.
            status = st.status("Thinking & calling tools…", expanded=True)

            def on_event(ev):
                if ev["type"] == "tool_call":
                    args = ", ".join(f"{k}={v}" for k, v in ev["args"].items())
                    status.write(f"🔧 **{ev['name']}**({args})")
                elif ev["type"] == "tool_result":
                    status.write(f"   ↳ {summarize_result(ev['result'])}")

            try:
                answer = run_conversation(
                    st.session_state.convo, model=MODEL, verbose=False, on_event=on_event
                )
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                answer = f"Something went wrong: {e}"
                status.update(label="Error", state="error")
            st.markdown(answer)
        st.session_state.chat_display.append(("assistant", answer))

st.sidebar.divider()
st.sidebar.caption("Data: Yahoo Finance (may be ~15 min delayed, end-of-day). "
                   "For learning only — not financial advice.")

# ---------------------------------------------------------------------------
# Top-level chat input — pinned to the bottom of the page (only docks there when
# called outside any tab/container). It queues the prompt and reruns; the Chat tab
# above picks it up and streams the answer. Note it's visible from every tab, so a
# question typed anywhere lands in the Chat tab.
# ---------------------------------------------------------------------------
if prompt := st.chat_input("💬 Ask the agent about an NSE stock or the market…"):
    st.session_state.pending_prompt = prompt
    st.rerun()
