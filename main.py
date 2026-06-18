"""
main.py — The entry point you actually run.

Usage:
    python main.py "Should I be interested in AAPL right now?"
    python main.py "How is Bitcoin (BTC-USD) doing this month?"

If you pass no question, it runs a default one so you can see it work immediately.
"""

import os
import sys

from dotenv import load_dotenv

from agent import run_agent

# Load OPENROUTER_API_KEY (and MODEL) from the .env file into the environment.
load_dotenv()

# Which model to use. Any OpenRouter model that supports tool/function calling works.
# Override it in .env (MODEL=...) without touching code. Browse ids at openrouter.ai/models.
MODEL = os.environ.get("MODEL", "google/gemini-2.5-flash")


def main() -> None:
    # Grab the user's question from the command line, or use a default.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Give me a quick read on Apple (AAPL) right now."

    print(f"Model: {MODEL}")
    print(f"Question: {question}\n")
    print("--- agent working ---")

    answer = run_agent(question, model=MODEL)

    print("\n--- final answer ---")
    print(answer)


if __name__ == "__main__":
    main()
