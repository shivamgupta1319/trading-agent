"""
universe.py — WHICH stocks the scanner looks at.

We hardcode the Nifty 50 (India's 50 largest, most liquid stocks) rather than
scraping it live. Why hardcode?
  - The list changes only ~twice a year (NSE reconstitutes it), so it's nearly static.
  - Live-scraping NSE's website is fragile (bot-blocking, layout changes) and would
    add failure points on the app's hot path.
  - A plain list is offline-resilient and dead simple to read.

Symbols are stored in Yahoo Finance form (with the '.NS' suffix) so they fetch
directly. Update this list by hand when the index reconstitutes (next time you do,
add the new date below).
"""

# Nifty 50 constituents — as of 2025 (reconstituted ~semiannually; verify if it matters).
NIFTY_50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TCS.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS",
    "WIPRO.NS",
]


def display_name(symbol: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE' for clean display."""
    return symbol.replace(".NS", "").replace(".BO", "")
