#!/usr/bin/env python3
"""
agent_sql_demo_min.py — Minimal NL→SQL agent demo (self-contained, classroom-ready)

Overview
--------
This script demonstrates a tiny, *engineered* agent loop that converts a natural-language
goal into a safe SQL query, runs it against MySQL in read-only mode, previews a few rows,
and decides whether to accept or revise the result.

The loop you will see:
    Planner  → proposes 1–3 concrete steps
    SQLWriter→ drafts a single SELECT statement using the live schema hint
    Executor → validates + runs the SQL, returns a small preview table
    Verifier → checks output contract (columns/aliases/emptiness/order) and decides:
               'accept' | 'revise' | 'stop'

This is a minimal reference for DSC 360. Use it as a template for your mini-capstone.

Assumptions and What You Need
-----------------------------
- You have a MySQL account from Lab 6 on the departmental server:
      host: cscdata.centre.edu
      database: gravity_books      (use this name, not 'gravity')
      user/password: your Lab 6 team credentials (read-only is recommended)
- You have an OpenAI account/key for the Chat Completions API.
  (If you do not, see “Adapting to Ollama” below; the swap is small.)
- Python 3.10+ and these packages:
      pip install openai mysql-connector-python

Quick Start
-----------
1) Export your OpenAI key in a plain-ASCII shell (no smart quotes):
      export OPENAI_API_KEY='sk-...'
   (The helper `_require_ascii_env` will fail fast if you paste curly quotes.)
2) Fill in your MySQL connection values in the config section (host/user/pass/db).
3) Run:
      python agent_sql_demo_min.py
4) At the prompt, type a goal, e.g.:
      Which books about SQL have never been ordered?
      For each customer, calculate the total amount they have spent on orders.
      List all books translated into more than two languages.

What You’ll See (Operator Panel)
--------------------------------
- Goal:        the natural-language request you typed
- Plan:        1–3 numbered steps the SQL writer will follow
- SQL:         a single SELECT produced by the agent (normalized)
- Preview:     a compact table of up to N rows (configurable)
- Verdict:     accept / revise / stop with a short reason
- Elapsed:     how long this round took (ms)

Safety Envelope (Guardrails)
----------------------------
- Read-only DB session where possible (session flags) + read-only credentials
- Static validator that only allows: WITH / SELECT / SHOW / DESCRIBE (DESC)
- Single-statement check (blocks multi-statement and semicolon-chaining)
- Small row preview; no full exports
- Low temperature for structured steps; “Return a single JSON object only.” hint
- All requests and decisions are printed for human oversight

Educational Intent (What to Notice)
-----------------------------------
- We “group by what we measure” (e.g., per **title** vs per **row**).
- We prefer `NOT EXISTS` / LEFT-anti patterns over `NOT IN (...)` with NULLs.
- Inner vs LEFT joins control whether zero-row entities appear in the preview.
- The verifier acts like unit tests for the query: required columns/aliases present,
  non-empty when implied, ordering + LIMIT when the goal demands “top-K,” etc.

Adapting to Ollama (no OpenAI account)
--------------------------------------
You can replace the tiny LLM wrapper with an Ollama call and keep the same interface:
- Keep `LLM.ask_json(system_prompt, user_prompt)` returning a Python dict.
- Use either the `ollama` Python package or a simple HTTP POST to localhost.
- Preserve the contract: system + user messages in → **single JSON object** out.
Everything else (planner/writer/executor/verifier/validator) stays the same.

Privacy & Data Notes
--------------------
- The schema hint (table/column names) and a *small* preview (few rows) are sent
  to the model for verification. Do not point this at sensitive datasets.
- For totally local processing, use a local model (see “Adapting to Ollama”).

Troubleshooting
---------------
- “Expected JSON …”: The model returned extra prose. The extractor grabs the first
  `{...}` block; keep temperature low and the “single JSON object only” instruction.
- Auth errors: Ensure `OPENAI_API_KEY` is ASCII (no “smart quotes”), and exported
  in the same shell where you run Python.
- MySQL “Access denied”: Confirm host/user/password and that your user can read
  the `gravity_books` schema on `cscdata.c_
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Literal

import mysql.connector as mc
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------------------
# Config: central knobs for model behavior and DB connection (edit here, not in the loop)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

DB_HOST = os.getenv("DB_HOST", "cscdata.centre.edu") # or localhost
DB_USER = os.getenv("DB_USER", "agent")              # team user_name: use Lab 6 credentials
DB_PASS = os.getenv("DB_PASS", "")                   # team password
DB_NAME = os.getenv("DB_NAME", "gravity_books")      # or gravity on Prof. Allen's machine

# Internal (leave as-is unless you need to tweak privately)
_DB_PORT = int(os.getenv("DB_PORT", "3306"))
_TEMPERATURE = 0.1
_MAX_ITERS = 5
_ROW_PREVIEW = 10

DESTRUCTIVE_VERB_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|LOCK|UNLOCK)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------
# Compact schema hint (edit if your schema differs)
# --------------------------------------------------------------------
SCHEMA_HINT = """\
address(address_id:int, street_number:varchar, street_name:varchar, city:varchar, country_id:int)
address_status(status_id:int, address_status:varchar)
author(author_id:int, author_name:varchar)
book(book_id:int, title:varchar, isbn13:varchar, language_id:int, num_pages:int, publication_date:date, publisher_id:int)
book_author(book_id:int, author_id:int)
book_language(language_id:int, language_code:varchar, language_name:varchar)
country(country_id:int, country_name:varchar)
cust_order(order_id:int, order_date:datetime, customer_id:int, shipping_method_id:int, dest_address_id:int)
customer(customer_id:int, first_name:varchar, last_name:varchar, email:varchar)
customer_address(customer_id:int, address_id:int, status_id:int)
order_history(history_id:int, order_id:int, status:varchar, change_date:datetime)
order_item(order_id:int, book_id:int, quantity:int, unit_price:decimal)
publisher(publisher_id:int, publisher_name:varchar)
shipping_method(shipping_method_id:int, shipping_method_name:varchar)
"""

# --------------------------------------------------------------------
# OpenAI Chat Completions (no Responses API; fewer moving parts)
# --------------------------------------------------------------------
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("Please `pip install openai` (>=1.0).")

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON strictly; if that fails, grab the first {...} block and parse it.
    Tolerates minor model drift (extra prose/code fences). Callers still validate.
    """
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Expected JSON, got:\n{text}")
        return json.loads(m.group(0))


def _require_ascii_env(varname: str) -> None:
    """
    Fail fast if an env var (e.g., OPENAI_API_KEY) contains non-ASCII (smart quotes).
    Prevents confusing auth/HTTP errors caused by pasted curly characters.
    """
    val = os.getenv(varname)
    if val is None:
        return
    try:
        val.encode("ascii")
    except UnicodeEncodeError:
        msg = (
            f"Environment variable {varname} contains non-ASCII characters.\n"
            f"This often happens if smart quotes or symbols were pasted (e.g., “sk-…”, bullets, en/em dashes).\n"
            f"Fix by re-exporting with plain ASCII, e.g.:\n"
            f"  export {varname}='sk-...'\n"
        )
        raise SystemExit(msg)

class LLM:
    """
    Minimal wrapper around OpenAI Chat Completions for JSON-in/JSON-out.
    Holds a client, default model, and temperature. Use `ask_json(...)` to send a
    system+user prompt and parse a single JSON object from the reply.
    """
    def __init__(self, model: str = OPENAI_MODEL, temperature: float = _TEMPERATURE):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def ask_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Send system+user messages and parse one JSON object from the reply.
        Appends 'Return a single JSON object only.' to reduce chatter.
        Returns: Python dict (caller should validate with Pydantic).
        Raises: JSON decode errors or OpenAI client errors propagate upward.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn a single JSON object only."},
            ],
        )
        content = resp.choices[0].message.content or ""
        return _extract_json(content)

# --------------------------------------------------------------------
# Pydantic models (v2) for structure
# --------------------------------------------------------------------
class PlanOut(BaseModel):
    """
    Planner output shape.
    steps: small, concrete actions the SQL writer can follow
    initial_sql: optional first attempt at a query
    """
    steps: List[str] = Field(default_factory=list)
    initial_sql: Optional[str] = None

class SQLDraft(BaseModel):
    """
    SQL writer output.
    sql: candidate SELECT statement (single statement, read-only)
    rationale: short explanation (optional; useful for debugging)
    """
    sql: str
    rationale: Optional[str] = ""

class Verdict(BaseModel):
    """
    Verifier decision.
    decision: 'accept' | 'revise' | 'stop'
    reason: one-sentence justification
    issues: optional list of specific problems to fix next
    """
    decision: Literal["accept", "revise", "stop"]
    reason: str
    issues: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------
# SQL safety and execution
# --------------------------------------------------------------------
ALLOWED_LEAD_TOKENS = ("WITH", "SELECT", "SHOW", "DESCRIBE", "DESC")
BLOCKLIST = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bREPLACE\b",
    r"\bDROP\b", r"\bCREATE\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bGRANT\b", r"\bREVOKE\b", r"\bSET\s+PASSWORD\b", r"\bCALL\b",
    r"\bUSE\s+\w+", r"\bLOCK\b", r"\bUNLOCK\b"
]

def _normalize_sql(sql: str) -> str:
    """
    Trim trailing semicolon and collapse internal whitespace.
    Normalization helps safety checks and log readability.
    """
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return re.sub(r"[ \t]+", " ", s)

def _is_single_statement(sql: str) -> bool:
    """
    Return True if `sql` contains at most one terminated statement.
    Accepts either a statement without ';' or exactly one ending with ';'.
    """
    s = sql.strip()
    if ";" not in s:
        return True
    return s.endswith(";") and s.count(";") == 1

def validate_sql_safe(sql: str) -> Tuple[bool, str]:
    """
    Static safety gate for read-only demos.
    Allows only WITH/SELECT/SHOW/DESCRIBE(DESC). Blocks DDL/DML and multi-statement input.
    Returns: (ok, message). Caller prints message when blocked.
    """
    s = _normalize_sql(sql)
    if not s:
        return False, "Empty SQL."
    if not _is_single_statement(s + ";"):
        return False, "Only a single statement is allowed."
    lead = re.split(r"\s+", s, maxsplit=1)[0].upper()
    if lead not in ALLOWED_LEAD_TOKENS:
        return False, f"Must start with WITH/SELECT/SHOW/DESCRIBE; got '{lead}'."
    for pat in BLOCKLIST:
        if re.search(pat, s, flags=re.IGNORECASE):
            return False, f"Blocked by safety rule: {pat}"
    return True, "OK"

def mysql_connect() -> mc.MySQLConnection:
    """
    Open a MySQL connection (autocommit, short timeout) and make a best-effort
    to mark the session READ ONLY and reduce long statements. Raises SystemExit with
    a friendly message if connection fails.
    """
    try:
        conn = mc.connect(
            host=DB_HOST, port=_DB_PORT, user=DB_USER, password=DB_PASS,
            database=DB_NAME, autocommit=True, connection_timeout=8
        )
        # best-effort session read-only + short timeout
        try:
            cur = conn.cursor()
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute("SET SESSION MAX_EXECUTION_TIME=500")  # cap time in ms
            cur.close()
        except Exception:
            pass
        return conn
    except Exception as e:
        msg = (
            f"\nFailed to connect to MySQL.\n"
            f"  host={DB_HOST} port={_DB_PORT}\n"
            f"  user={DB_USER} db={DB_NAME}\n"
            f"Error: {e}\n"
            f"Tips: Is mysqld running? Credentials correct? Can you `mysql -h {DB_HOST} -u {DB_USER} -p`?\n"
        )
        raise SystemExit(msg)


def mysql_execute_preview(conn, sql: str, max_rows: int = _ROW_PREVIEW) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """
    Execute a read-only SELECT and return a small preview.
    Uses a buffered dict cursor. Returns (columns, first `max_rows` rows, total_count*).
    *total_count relies on cursor.rowcount; if -1, we fall back to len(rows).
    """
    try:
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute(sql)
        columns = list(cur.column_names) if cur.column_names else []
        rows = cur.fetchmany(size=max_rows) if columns else []
        total = cur.rowcount if cur.rowcount != -1 else len(rows)
        cur.close()
        return columns, rows, total
    except Exception as e:
        raise RuntimeError(f"MySQL execution error: {e}")


def format_table(columns: List[str], rows: List[Dict[str, Any]], max_width: int = 96) -> str:
    """
    Render a compact ASCII table for previews.
    Truncates wide cell text with an ellipsis and fits to `max_width`.
    """
    if not columns:
        return "(no columns)"
    values = [[str(r.get(col, "")) for col in columns] for r in rows]
    widths = [max(len(col), *(len(v[i]) for v in values)) if values else len(col) for i, col in enumerate(columns)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def row_line(vals: List[str]) -> str:
        cells = []
        for i, v in enumerate(vals):
            s = v
            if len(s) > max_width:
                s = s[: max_width - 1] + "…"
            cells.append(f" {s:<{widths[i]}} ")
        return "|" + "|".join(cells) + "|"
    lines = [sep, row_line(columns), sep]
    for vs in values:
        lines.append(row_line(vs))
    lines.append(sep)
    return "\n".join(lines)

# --------------------------------------------------------------------
# Live schema introspection (minimal pass)
# --------------------------------------------------------------------
def build_schema_hint(conn, max_tables: int = 80, max_cols: int = 12) -> str:
    """
    Introspect the current DB and build a one-line-per-table schema hint:
    e.g.,  book(book_id:int, title:varchar, ...).  Keeps output bounded for prompts.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT TABLE_NAME, TABLE_TYPE "
            "FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "ORDER BY TABLE_NAME"
        )
        tbls = cur.fetchall()
        cur.close()
    except Exception as e:
        return "(schema unavailable: " + str(e) + ")"

    lines = []
    count = 0
    for name, ttype in tbls:
        if count >= max_tables:
            lines.append("…")
            break
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COLUMN_NAME, DATA_TYPE "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s "
                "ORDER BY ORDINAL_POSITION",
                (name,),
            )
            cols = cur.fetchall()
            cur.close()
        except Exception as e:
            cols = []
        parts = []
        for i, (col, dtype) in enumerate(cols):
            if i >= max_cols:
                parts.append("…")
                break
            parts.append(f"{col}:{dtype}")
        lines.append(f"{name}({', '.join(parts)})")
        count += 1
    return "\n".join(lines)

# --------------------------------------------------------------------
# Prompts for LLM
# --------------------------------------------------------------------
PLANNER_SYSTEM = """\
You are the Planner in a tiny SQL agent demo. Be concise.
Return: {"steps": [...], "initial_sql": "..." | null}
Only read-only SQL (WITH/SELECT/SHOW/DESCRIBE), one statement.
"""

PLANNER_USER_TMPL = """\
Goal:
{goal}

Schema Hint:
{schema}

If not obvious, set initial_sql = null.
"""

SQL_WRITER_SYSTEM = """\
You are the SQL Writer. Produce ONE safe read-only MySQL 8+ statement.
Allowed: WITH, SELECT, SHOW, DESCRIBE (one statement only).
General best practices (apply when relevant):
- Prefer NOT EXISTS (or LEFT JOIN ... IS NULL) for anti-joins instead of NOT IN (subquery).
- Use COALESCE with aggregates when nulls are possible (e.g., COALESCE(SUM(...), 0)).
- For “top N”, include ORDER BY and LIMIT.
- When aggregating across multiple rows of the same real-world entity, group by the entity’s natural key(s), not a surrogate unique id. Example: if counting distinct languages across editions that share a title, GROUP BY title (not book_id).
Return: {"sql":"...", "rationale":"..."}
"""

SQL_WRITER_USER_TMPL = """\
Goal:
{goal}

Schema Hint:
{schema}

Plan Steps:
{steps}

Prior Attempt:
{prior_sql}

Observed Issue (if any):
{issue}
"""

VERIFIER_SYSTEM = """\
You are the Verifier. Decide if the SQL meets the goal given the sample.
Rules (generic, not schema-specific):
- Do NOT accept an empty sample when the goal implies rows; prefer "revise". Use "stop" only if the schema clearly cannot support the goal.
- If the goal requests specific columns (and/or aliases), require they appear and be non-NULL in sample rows.
- If “top N” or ordering is implied, require ORDER BY and LIMIT (or equivalent).
- If the SQL counts DISTINCT of a column but GROUP BY includes a unique row identifier (e.g., a surrogate primary key), flag as likely incorrect and ask to group by the natural key(s) for the entity being reported.
Return: {"decision":"accept|revise|stop", "reason":"...", "issues":[...]}
"""

VERIFIER_USER_TMPL = """\
Goal:
{goal}

Schema Hint:
{schema}

SQL:
{sql}

Sample (first rows; may be empty):
{sample}
"""

# --------------------------------------------------------------------
# Agent core (Planner → SQL Writer → Verifier)
# --------------------------------------------------------------------
def _steps_to_text(steps: List[str]) -> str:
    """
    Numbered bullet list for planner steps, suitable for display and prompting.
    """
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

def ask_plan(llm: LLM, goal: str, schema: str) -> PlanOut:
    """
    Call the Planner with the user goal and schema hint.
    Returns a validated PlanOut; on malformed JSON, returns an empty plan.
    """
    data = llm.ask_json(PLANNER_SYSTEM, PLANNER_USER_TMPL.format(goal=goal, schema=schema))
    try:
        return PlanOut.model_validate(data)
    except ValidationError:
        return PlanOut(steps=[], initial_sql=None)

def ask_writer(llm: LLM, goal: str, schema: str, steps_txt: str, prior_sql: Optional[str], issue: str) -> SQLDraft:
    """
    Call the SQL Writer with the goal, schema, planner steps, prior SQL, and latest issue.
    Returns a validated SQLDraft; on malformed JSON, falls back to extracting 'sql' if present.
    """
    data = llm.ask_json(
        SQL_WRITER_SYSTEM,
        SQL_WRITER_USER_TMPL.format(
            goal=goal, schema=schema, steps=steps_txt or "(none)",
            prior_sql=prior_sql or "null", issue=issue or "(none)",
        ),
    )
    try:
        return SQLDraft.model_validate(data)
    except ValidationError:
        sql = data.get("sql") if isinstance(data, dict) else None
        if not isinstance(sql, str):
            raise RuntimeError(f"SQL Writer returned invalid JSON: {data}")
        return SQLDraft(sql=sql, rationale=data.get("rationale", ""))

def ask_verdict(llm: LLM, goal: str, schema: str, sql: str, sample: Dict[str, Any]) -> Verdict:
    """
    Call the Verifier with the goal, schema, candidate SQL, and a sample (cols + up to 5 rows).
    Enforces the output contract (columns/aliases present, non-empty when implied, ordering/limit when needed).
    Returns a validated Verdict; otherwise asks for a revise.
    """
    data = llm.ask_json(
        VERIFIER_SYSTEM,
        VERIFIER_USER_TMPL.format(goal=goal, schema=schema, sql=sql, sample=json.dumps(sample, default=str)),
    )
    try:
        return Verdict.model_validate(data)
    except ValidationError:
        return Verdict(decision="revise", reason="Unclear JSON verdict.", issues=["Return valid JSON."])

def solve_problem(llm, conn, goal, schema, max_iters=_MAX_ITERS):
    """
    Orchestrate the loop: Planner → SQL Writer → (validator+executor+preview) → Verifier.

    Flow:
      1) Refuse destructive goals up front (read-only demo).
      2) Ask Planner for steps and optional initial SQL; display steps.
      3) For up to `max_iters`:
         - Ask Writer for SQL (with latest issues if any).
         - Run `validate_sql_safe` → block if multi-stmt/unsafe.
         - Execute and show a small preview (or '(no rows)').
         - If preview is empty but rows seem implied, prompt a gentle retry once or twice.
         - Ask Verifier; if decision is accept/stop → return result; else loop with issues.

    Returns: dict with {sql, decision, reason, issues, columns, rows, rowcount, steps}.
    """

    # Refuse destructive goals (minimal string check: not hardened)
    if DESTRUCTIVE_VERB_RE.search(goal):
        msg = "Refused: this demo is read-only and cannot modify data."
        print("\n--- Refusal ---")
        print(msg)
        return {
            "sql": "",
            "decision": "stop",
            "reason": msg,
            "issues": [],
            "columns": [],
            "rows": [],
            "rowcount": 0,
            "steps": [],
        }

    # As an initial step, ask the planner for steps that will lead to a solution
    # Note: We don't attempt to modify the plan during the agent loop, but could if we liked
    print("\n--- Planner ---")

    # Low temperature + “single JSON object only” → stable, parseable replies
    plan = ask_plan(llm, goal, schema)
    steps_txt = _steps_to_text(plan.steps)
    print(steps_txt or "(no steps)")
    candidate_sql = plan.initial_sql
    prior_issue = ""

    # Initiate agent loop with fixed limit on number of iterations so we don't get stuck
    for it in range(1, max_iters + 1):

        # Ask the Writer for SQL, passing it state information that may be useful
        print(f"\n--- Iteration {it}: SQL Writer ---")
        draft = ask_writer(llm, goal, schema, steps_txt, candidate_sql, prior_issue)
        candidate_sql = _normalize_sql(draft.sql)
        print("Proposed SQL:\n", candidate_sql)

        # Pass the generated SQL to the validator for safety inspection
        ok, msg = validate_sql_safe(candidate_sql)
        if not ok:
            print(f"Blocked by validator: {msg}")
            prior_issue = f"Blocked: {msg}"
            continue

        try:
            # Execute the SQL, retrieving a small preview of the result set
            cols, rows, total = mysql_execute_preview(conn, candidate_sql, max_rows=_ROW_PREVIEW)
            print("\nPreview:")
            if rows:
                print(format_table(cols, rows))
                print(f"(previewed {len(rows)} rows; total reported/estimated: {total})")
            else:
                print("(no rows returned)")

            # Heuristic: if the goal implies rows, n_preview == 0 triggers a gentle retry or revision
            if not rows:
                prior_issue = (
                    "Sample was empty but the goal appears to imply rows. "
                    "Consider checking join keys, relaxing filters, or verifying the schema supports the goal."
                )
                continue

        except Exception as e:
            # If we encounter an error during execution, add that to
            # the state information (prior_issue) for the next iteration
            err = str(e)
            print(f"Execution error: {err}")
            prior_issue = f"Execution error: {err}"
            continue

        # Call the Verifier to determine what state we are in (solved, no solution, continue)
        print("\n--- Verifier ---")
        verdict = ask_verdict(llm, goal, schema, candidate_sql, {"columns": cols, "rows": rows[:5]})
        print(f"Decision: {verdict.decision}\nReason: {verdict.reason}")
        if verdict.issues:
            print("Issues:", "; ".join(verdict.issues))

        if verdict.decision in ("accept", "stop"):
            return {
                "sql": candidate_sql,
                "decision": verdict.decision,
                "reason": verdict.reason,
                "issues": verdict.issues,
                "columns": cols,
                "rows": rows,
                "rowcount": total,
                "steps": plan.steps,
            }

        prior_issue = verdict.reason if not verdict.issues else "; ".join(verdict.issues)

    # Return the result object
    return {
        "sql": candidate_sql or "",
        "decision": "stop",
        "reason": "Reached max iterations without acceptance.",
        "issues": [prior_issue] if prior_issue else [],
        "columns": [],
        "rows": [],
        "rowcount": 0,
        "steps": plan.steps,
    }

# --------------------------------------------------------------------
# REPL (no CLI). Type a goal; get a run. /exit or Ctrl-D to quit.
# --------------------------------------------------------------------
def main() -> None:
    """Tiny REPL for classroom demos.
    - Verifies OpenAI env vars for ASCII safety and presence.
    - Connects to MySQL, builds a live schema hint, constructs the LLM.
    - Loop: read a goal (or /exit), run `solve_problem`, print operator-panel output:
      Planner steps, proposed SQL, preview table, verdict, elapsed time.
    """
    # Make sure we have an OpenAI API key
    # Ensure OpenAI-related envs are ASCII-only (headers must be ASCII)
    _require_ascii_env("OPENAI_API_KEY")
    _require_ascii_env("OPENAI_ORGANIZATION")
    _require_ascii_env("OPENAI_PROJECT")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # Initialize Database Connection
    print(f"Connecting to MySQL ({DB_HOST}:{_DB_PORT}, db={DB_NAME}) …")
    conn = mysql_connect()
    # Build a live, compact schema hint from MySQL (read-only)
    live_schema = build_schema_hint(conn)
    print("\n[Schema hint loaded from DB]\n")
    llm = LLM()

    print("\nDSC 360 — Minimal SQL Agent Demo (type /exit to quit)\n")
    # print(live_schema + "\n")
    print("Example: List the top 10 languages and number of books in each.\n")

    # REPL (Read-Execute-Print Loop) - for debugging and demo
    try:
        while True:
            # Get goal (i.e., natural language database query) from user or quit
            try:
                goal = input("Goal> ").strip()
            except EOFError:  # User presses Ctrl-D --> EOF
                print()
                break
            if not goal:
                continue
            if goal.lower() in {"/exit", "\\exit", "quit", ":q"}:
                break

            # Call the solver (initiate agent loop) and measure latency
            t0 = time.time()
            result = solve_problem(llm, conn, goal, live_schema, _MAX_ITERS)
            dt = time.time() - t0

            # Display result to the user for demo
            print("\n=== RESULT ===")
            print("SQL:", result["sql"] or "(none)")
            print("Decision:", result["decision"])
            print("Reason:", result["reason"])
            if result["issues"]:
                print("Issues:", "; ".join(result["issues"]))
            if result["columns"] and result["rows"]:
                print("\nPreview:")
                print(format_table(result["columns"], result["rows"]))
            print(f"\nElapsed: {dt:.2f}s\n")

    # Close the database connection
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
