"""
agent.py — The "brain loop". THIS is the part that makes it an *agent*.

Everything important about agents is in the run_agent() loop below. Read it slowly:

    1. Send the conversation + the list of tools to the model.
    2. The model replies with EITHER:
         (a) tool calls  -> "please run get_quote('AAPL') for me"
         (b) a final text answer -> "done, here's my analysis"
    3. If it asked for tools, WE run them, append the results, and loop again.
    4. If it gave a final answer, we stop and return it.

A normal chatbot does step 1 once. An agent repeats the loop, so the model can
gather information across multiple steps and decide its own next move. That loop
is the whole idea.
"""

import json
import os

from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


# The system prompt sets the agent's role, rules, and personality. It's the most
# powerful knob you have for steering behavior — change this and the agent changes.
SYSTEM_PROMPT = """You are a careful technical-analysis assistant for the INDIAN
stock market (NSE). You support three timeframes, and most tools take a `timeframe`:
  - 'swing' (DEFAULT): daily candles, holds of days–weeks.
  - 'intraday': 15-min candles, same-day trades (data may be delayed ~15 min).
  - 'positional': weekly candles, holds of months+.
Pick the timeframe from the user's intent — if they say "day trade / intraday / today's
move" use intraday; "long-term / investing / for months" use positional; otherwise swing.
The same indicators and strategies apply to every timeframe (an EMA50 is just 50 bars).

Symbols: Indian NSE tickers like RELIANCE, TCS, INFY, HDFCBANK. The tools accept bare
tickers and add the '.NS' suffix automatically.

Your job: when the user names a stock, use your tools to gather evidence — a quote,
recent history, and especially the technical indicators (get_indicators) — then give a
short, structured read:
  - Current price and recent move.
  - Trend (from EMA20/50), momentum (from RSI and MACD), and volatility (from ATR).
  - A balanced swing-trading view: what the setup looks like and what would invalidate it.

Tool guidance:
  - For "how does X look / what do indicators say" → use get_indicators.
  - For "is there a trade / where do I enter & exit / what's the setup" → use
    analyze_setup. If the user mentions their capital, pass it so you also get a
    position size. Report entry, stop, target and risk:reward clearly.
    Three strategies are available: 'trend' (buy established uptrends), 'breakout'
    (buy new highs on volume), and 'mean_reversion' (buy oversold dips in an
    uptrend). If the user doesn't specify, default to 'trend'; if they describe a
    style, pick the matching strategy. You can compare strategies on the same stock.
  - For "how many shares should I buy" with a user-supplied entry & stop → use
    position_size.
  - For "does this strategy work / is it reliable / what's the win-rate" → use
    backtest_strategy. When you present a fresh setup and the user cares whether it's
    trustworthy, back it up by also backtesting the strategy on that stock and quoting
    the win-rate and expectancy.
  - For "scan the market / best setups today / find me candidates across Nifty 50" →
    use scan_market. Present the top results as a clear ranked list with entry, stop,
    target and the backtested win-rate/expectancy. Flag any 'low confidence' rows
    (few backtest trades) and mention how many of the 50 actually triggered.

Rules:
  - ALWAYS back claims with data you actually fetched. Never invent numbers.
  - When a setup is NOT triggered, explain which rules failed (the reasons are
    provided) — a clear "no trade, because…" is a valid, useful answer.
  - A stop-loss and position size are not optional extras — always surface them
    when you present a trade.
  - Free data may be delayed ~15 minutes and is end-of-day — fine for swing, not intraday.
  - You are educational decision-support ONLY, not a financial advisor. End every
    analysis with a one-line reminder that this is not financial advice.
  - If a symbol looks invalid, say so plainly.
"""


def build_client() -> OpenAI:
    """Create the model client, pointed at OpenRouter (OpenAI-compatible API)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY. Copy .env.example to .env and add your key."
        )
    # OpenRouter exposes an OpenAI-compatible endpoint, so we reuse the openai SDK
    # and just point it at OpenRouter's base URL.
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def new_conversation() -> list:
    """Start a fresh conversation — just the system prompt, no turns yet.

    Keep this list around (e.g. in the GUI's session state) and pass it to
    run_conversation each turn so the agent REMEMBERS the previous exchange.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_conversation(messages: list, model: str, max_steps: int = 6, verbose: bool = True,
                     on_event=None) -> str:
    """Run the agent loop on an EXISTING conversation and return the final answer.

    `messages` is the full running conversation (system prompt + every prior turn +
    the latest user message). It is mutated in place — the model's reply and any tool
    calls/results are appended — so after this returns, `messages` is ready for the
    next user turn. THIS is what makes multi-turn chat work: the agent can see what it
    said before, so a follow-up like "yes, go ahead" makes sense.

    `on_event` is an optional callback that lets a UI watch the loop in real time. It's
    called with small dicts:
        {"type": "tool_call",   "name": str, "args": dict}
        {"type": "tool_result", "name": str, "result": any}
    The Streamlit chat uses this to stream "🔧 calling get_indicators…" as it happens.

    max_steps is a safety cap so a confused model can't loop forever.
    """
    client = build_client()

    for step in range(max_steps):
        # ---- Step 1: ask the model what to do next ----
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,   # <- this is what makes tool-calling possible
            max_tokens=1024,      # cap the reply length (controls cost; also avoids
                                  # over-reserving credits on a small balance)
        )
        msg = response.choices[0].message

        # Record the model's reply in the conversation. We rebuild it as a plain
        # dict so the structure (and tool_calls) is explicit and easy to inspect.
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # ---- Step 2: did the model ask to run tools, or is it done? ----
        if not msg.tool_calls:
            # No tool calls => this is the final answer. Stop the loop.
            return msg.content or "(the model returned no text)"

        # ---- Step 3: run every requested tool and feed results back ----
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)  # model sends args as JSON text

            if verbose:
                print(f"  [step {step + 1}] model calls: {name}({args})")
            if on_event:
                on_event({"type": "tool_call", "name": name, "args": args})

            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                result = func(**args)  # <- OUR code runs the real function here

            if verbose:
                print(f"            -> {result}")
            if on_event:
                on_event({"type": "tool_result", "name": name, "result": result})

            # A "tool" message hands the result back to the model, tagged with the
            # tool_call_id so the model knows which call it answers.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
        # Loop again — now the model can see the tool results and decide what's next.

    return "Stopped: hit the max step limit without a final answer."


def run_agent(user_message: str, model: str, max_steps: int = 6, verbose: bool = True) -> str:
    """Single-turn convenience wrapper: ask one question, get one answer.

    Used by the CLI (main.py). For multi-turn chat, use new_conversation() +
    run_conversation() instead so the agent keeps its memory between turns.
    """
    messages = new_conversation()
    messages.append({"role": "user", "content": user_message})
    return run_conversation(messages, model, max_steps=max_steps, verbose=verbose)
