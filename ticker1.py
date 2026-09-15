"""
Example: extracting structured info from noisy tweets with an LLM (Ollama) + Pydantic.

Goal (what the program does):
  From casual posts, extract a tiny JSON record like:
    {"ticker": "KO", "price": 58.2}

Why this matters:
  LLMs are "squishy"—they may answer in sentences. We need *structured* data
  (a dictionary with known keys/types) so the rest of our code can rely on it.
  We combine:
    - an LLM to read the messy text,
    - a schema to nudge the LLM into JSON,
    - and Pydantic to *enforce* the rules (types, bounds) before we use the data.

Requirements:
  - pip install "pydantic>=2"
  - Install and run Ollama locally (https://ollama.com)
  - Pull a model:  ollama pull gemma3:12b

Example Output:

  % python ticker1.py

  POST: Starter in Exon Mobil at 116 today. Fueling the retirement RV. ⛽️
  EXTRACTED: {'ticker': 'XOM', 'price': 116.0}

  POST: Berkshire B shares added @ $443 — slow and steady.
  EXTRACTED: {'ticker': 'BRK.B', 'price': 443.0}

  POST: Picked up iphone makers stock for 192.5 — long term. How bout them Apple!
  EXTRACTED: {'ticker': 'AAPL', 'price': 192.5}

  POST: Walmart at sixty-four and three-quarters — steady and boring wins. 🛒
  EXTRACTED: {'ticker': 'WMT', 'price': 64.75}
"""

# Enable forward references in type hints (a harmless, modern default)
from __future__ import annotations

# Built-in module for working with JSON (serialize/deserialize text ↔ Python objects)
import json

# Regular expression (regex) library — used for pattern matching (e.g., stock tickers)
import re

# Typing tools for type hints and structured data
from typing import Any, Dict

# Ollama client — lets Python send messages to a local LLM instance
from ollama import chat

# Pydantic — validates and structures untrusted data using Python classes
from pydantic import BaseModel, ValidationError, field_validator


# ---------- Pydantic model (our "contract" for structured data) ----------

# We define a *class* to describe what a valid stock trade looks like.
# Think of a class as a blueprint for a new *type* of object.
#   Example: int and str are built-in types; StockPrice will be our custom type.
#
# `BaseModel` (from Pydantic) is the *parent class* that we inherit from.
# Inheritance means our new class automatically gets BaseModel’s features:
#   - automatic type checking
#   - conversion from JSON/dict to Python objects
#   - built-in validation and error reporting
#
# When we create a StockPrice from a dictionary, Pydantic will:
#   (1) check that all required fields exist,
#   (2) convert types if possible (e.g., "58.2" → 58.2),
#   (3) run any field validators we define,
#   (4) return a clean, validated object (or raise an error).
#
# About the decorators below:
#   - A *decorator* (the @something syntax) attaches extra behavior to a function.
#     Example: @field_validator means “run this function when checking that field.”
#   - The `@classmethod` decorator makes the function belong to the *class*,
#     not to any single instance. Its first argument, `cls`, refers to the class itself.
#   - The combination of these lets us write small functions that clean or check
#     individual fields automatically whenever we create a StockPrice.

class StockPrice(BaseModel):
    """
    Represents one structured record extracted from a social-media post:
      - ticker (string): the stock symbol (e.g., 'KO', 'AAPL', 'BRK.B')
      - price  (float):  the purchase price in USD
    """
    ticker: str
    price: float
    
    # --- Validators for StockPrice ----------------------------------------------
    # Decorator quick intro:
    #   - A *decorator* (the @something syntax) attaches extra behavior to a function.
    #   - @field_validator tells Pydantic to run this function when checking a field.
    #   - @classmethod makes it a class method; the first parameter is `cls` (the class).

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, v: Any) -> str:
        # Normalize *before* type conversion so " aapl " → "AAPL"
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("ticker")
    @classmethod
    def check_ticker_pattern(cls, ticker: str) -> str:
        """
        Validate the ticker using a simplified pattern for this course:
        1–5 capital letters, optionally '.X' with 1–2 capital letters (e.g., BRK.B).
        This is a simplified instructional pattern—real exchanges have more cases.
        """

        # --- Regex primer -------------------------------------------------------
        # Regular expressions (regex) describe text patterns compactly.
        # Pattern we use (a simplified course rule for U.S.-style tickers):
        #   ^                start of string
        #   [A-Z]{1,5}       1–5 capital letters
        #   (?:\.[A-Z]{1,2})? optional group: dot + 1–2 capital letters (e.g., ".B")
        #   $                end of string
        # Examples:  "AAPL" ✅   "BRK.B" ✅   "brk.b" ❌   "APPLE1" ❌
        # Learn more:
        #   - Python docs: https://docs.python.org/3/library/re.html
        #   - Visualizer:  https://regex101.com/
        #   - Tutorial:    https://realpython.com/regex-python/
        # -----------------------------------------------------------------------

        pattern = r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$"
        if not re.fullmatch(pattern, ticker):
            raise ValueError(
                "Ticker must be 1–5 capital letters, optionally '.X' (e.g., KO, AAPL, BRK.B)."
            )
        return ticker

    @field_validator("price")
    @classmethod
    def validate_price_bounds(cls, price: float) -> float:
        """
        Price must be positive and not absurdly large (≤ $10M covers outliers like BRK.A).
        """
        p = float(price)  # in case the model returns "192.5" as a string
        if not (0.0 < p <= 10_000_000):
            raise ValueError("Stock price out of expected range (0, 10,000,000].")
        return p


# Generate JSON Schema *from* the model so rules stay in sync.
# We hand this schema to the LLM via Ollama so it *tries* to return exactly this shape.
# (Requires a relatively recent Ollama build that supports `format=` with a schema dict.)
JSON_SCHEMA: Dict[str, Any] = StockPrice.model_json_schema()


# ---------- Prompt (how we talk to the LLM) ----------
# chat() expects a *list of messages* with roles.
# - SYSTEM: high-level rules the assistant should always follow
# - USER: the actual task
# Tip: Be explicit that you want raw JSON and *nothing else*.
SYSTEM_MSG = """You are an information extractor for casual social-media posts about stock purchases.
Return ONLY a compact JSON object with exactly two fields:

- "ticker": US exchange ticker (uppercase). Allow class shares like "BRK.B".
- "price": the purchase price in USD as a number (no currency symbols).

If the ticker is not explicitly stated, infer it from context (e.g., company name or product).
No prose, no markdown, no code fencing — JSON only.
"""

USER_TEMPLATE = """Extract the trade from this post:

{message}

Return JSON like: {{"ticker": "KO", "price": 58.20}}
"""


def extract_trade(message: str, model: str = "gemma3:12b") -> StockPrice:
    """
    What this function does, step by step:
      1) Ask the LLM to produce strictly-structured JSON (we pass a schema to guide it).
      2) Parse the returned text as JSON (a Python dict).
      3) Validate with Pydantic to *guarantee* it's correct before we use it.

    Why temperature=0?
      We want determinism for grading, debugging, and reproducibility.
    """
    resp = chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},  # global variable defined above
            {"role": "user", "content": USER_TEMPLATE.format(message=message)}, # substitutes for {message} in variable USER_TEMPLATE defined above
        ],
        format=JSON_SCHEMA,            # Ask for structured output that matches our schema
        options={"temperature": 0},    # Deterministic extraction for labs/grading
    )

    # Ollama returns {"message": {"role": "...", "content": "<JSON string>"}}
    content = resp["message"]["content"]

    # Parse + validate in one step (Pydantic v2):
    # - parses the JSON string
    # - enforces the StockPrice schema (types, bounds, validators)
    try:
        return StockPrice.model_validate_json(content)
    except ValidationError as ve:
        # Display an error message if the LLM failed to give us valid JSON string
        raise ValueError(f"Invalid structured output: {ve}") from ve


# ---------- Tiny demo ----------
def main() -> None:
    print("Demo — extracting (ticker, price) using model: gemma3:12b")

    # These examples are intentionally noisy to stress inference and parsing:
    examples = [
        "Starter in Exon Mobil at 116 today. Fueling the retirement RV. ⛽️",  # Exxon Mobil → XOM
        "Berkshire B shares added @ $443 — slow and steady.",                  # Berkshire Hathaway B → BRK.B
        "Picked up iphone makers stock for 192.5 — long term. How bout them Apple!",  # Apple → AAPL
        "Walmart at sixty-four and three-quarters — steady and boring wins. 🛒",      # 64.75
    ]

    for text in examples:
        print("\nPOST:", text)
        try:
            trade = extract_trade(text, model="gemma3:12b")
            # model_dump() converts our object back to a plain dict for easy printing/JSON
            print("EXTRACTED:", trade.model_dump())
        except Exception as e:
            print("ERROR   :", e)


if __name__ == "__main__":
    main()
