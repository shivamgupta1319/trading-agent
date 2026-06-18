"""
risk.py — The RISK MANAGEMENT math. This is what separates trading from gambling.

A good trade idea is worthless without three numbers: where you get OUT if you're
wrong (stop-loss), where you take profit (target), and HOW MUCH to buy so a single
loss can't hurt you (position size). The pros obsess over this far more than entries.

All functions here are pure arithmetic — no data fetching, no model. Long-only for
now (we trade in the direction of "buy low, sell higher"); shorts come later.
"""

import math


# ---------------------------------------------------------------------------
# Stop-loss placement — "if price hits here, I was wrong, get out."
# ---------------------------------------------------------------------------
def atr_stop(entry: float, atr: float, atr_mult: float = 2.0) -> float:
    """Volatility-based stop: place the stop a few ATRs below entry.

    ATR = the stock's typical daily range. A volatile stock needs a WIDER stop so
    normal noise doesn't knock you out; a calm stock can use a tighter one. Using
    ATR makes the stop adapt to each stock automatically. 2x ATR is a common choice.
    """
    return round(entry - atr_mult * atr, 2)


def swing_low_stop(recent_lows, buffer_pct: float = 0.005) -> float:
    """Structure-based stop: just below the recent swing low.

    The idea: if price falls below the lowest point of the recent pullback, the
    uptrend structure is broken. We subtract a small buffer (0.5%) so we're not
    sitting exactly on the obvious level everyone watches.
    """
    low = float(min(recent_lows))
    return round(low * (1 - buffer_pct), 2)


# ---------------------------------------------------------------------------
# Target and risk:reward — "is this trade even worth taking?"
# ---------------------------------------------------------------------------
def target_from_rr(entry: float, stop: float, rr_target: float = 2.0) -> float:
    """Set a profit target as a multiple of the risk.

    If you risk ₹10 to make ₹20, that's a 2:1 reward-to-risk trade. Demanding at
    least 2:1 means you can be right less than half the time and still come out ahead.
    """
    risk_per_share = entry - stop
    return round(entry + rr_target * risk_per_share, 2)


def risk_reward(entry: float, stop: float, target: float) -> float:
    """Compute the actual reward-to-risk ratio of a trade.

    RR = (potential profit per share) / (potential loss per share).
    Higher is better; most swing traders want >= 1.5 or 2.0.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        # Stop must be below entry for a long. Guard against bad inputs.
        return 0.0
    reward_per_share = target - entry
    return round(reward_per_share / risk_per_share, 2)


# ---------------------------------------------------------------------------
# Position sizing — THE most important risk control.
# ---------------------------------------------------------------------------
def position_size(entry: float, stop: float, capital: float, risk_pct: float = 1.0) -> dict:
    """How many shares to buy so that hitting your stop loses only `risk_pct` of capital.

    The golden rule of survival: never risk more than a small slice (1-2%) of your
    account on one trade. Then a losing streak is survivable.

    Math:
      risk_amount    = capital * risk_pct%        e.g. 1% of ₹1,00,000 = ₹1,000
      per_share_risk = entry - stop               loss per share if stopped out
      qty            = risk_amount / per_share_risk   (rounded DOWN — never over-risk)

    We also report capital_deployed (qty * entry) and cap qty if it would exceed
    the capital you actually have.
    """
    per_share_risk = entry - stop
    if per_share_risk <= 0:
        return {"error": "Stop-loss must be below entry for a long trade."}

    risk_amount = capital * (risk_pct / 100)
    qty = math.floor(risk_amount / per_share_risk)

    capped = False
    if qty * entry > capital:
        # Can't afford the full risk-based size — buy as many as the cash allows.
        qty = math.floor(capital / entry)
        capped = True

    return {
        "shares": qty,
        "risk_per_share": round(per_share_risk, 2),
        "max_loss_at_stop": round(qty * per_share_risk, 2),
        "capital_deployed": round(qty * entry, 2),
        "risk_pct_used": risk_pct,
        "capital_capped": capped,  # True if cash, not risk %, limited the size
    }
