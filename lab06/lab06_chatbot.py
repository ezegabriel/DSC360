#!/usr/bin/env python3
# Lab06 — Gravity Books Chatbot
# - LLM->SQL always
# - LLM for final NL only when result is a single metric (e.g., COUNT)
# - Deterministic bullets/table for lists (Top-N)
# Files: schema.txt
# Run: python3 lab06_chatbot.py


import os, re, sys, json, urllib.request, datetime


# --- slash-command handling ---
HELP_TEXT = """
Gravity Books — commands
 /help       Show this help
 /exit       Quit
 /quit       Quit


Ask in English or paste raw SQL. Examples:
 • top 5 authors with most sales
 • top 10 publishers by revenue in 2021
 • items for order 1234
 • SELECT COUNT(*) AS n FROM book;
""".strip()


def parse_slash_command(s: str):
   """
   Return a normalized command name if the line starts with '/', else None.
   Accepts variants like '/help', '/HELP', '/help\\', '/help   '.
   """
   s = s.strip()
   if not s.startswith("/"):
       return None
   m = re.match(r"^/([A-Za-z?]+)", s)  # grab letters like 'help', 'exit', 'quit', or '?'
   return m.group(1).lower() if m else ""  # '' means a bare '/' (treated as unknown)






# ===== EDIT YOUR DB CREDS =====
DB_HOST = "cscdata.centre.edu"
DB_PORT = 3306
DB_NAME = "gravity_books"
DB_USER = "db_agent_b1"
DB_PASS = "Boomerang26!"
# ==============================


MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_LIMIT = 200
SMALL_BULLET_THRESHOLD = 20   # <= this => bullets; otherwise print table
SCHEMA_FILE = "schema.txt"
AUDIT_FILE = "audit.jsonl"


DISALLOWED = {"DELETE","UPDATE","INSERT","DROP","ALTER","TRUNCATE","CREATE","GRANT","REVOKE"}
DEICTIC = ("here","this","these","there","it")


def die(msg): print("FATAL:", msg, file=sys.stderr); sys.exit(1)
def now_iso(): return datetime.datetime.now().isoformat(timespec="seconds")


# ---- schema names (tables/views) ----
if not os.path.exists(SCHEMA_FILE): die("schema.txt not found.")
SCHEMA = open(SCHEMA_FILE, "r", encoding="utf-8").read().strip()


def allowed_from_schema(text: str):
   names, sect = set(), None
   for line in text.splitlines():
       s = line.strip()
       if not s: continue
       if s.startswith("[") and s.endswith("]"):
           sect = s.lower(); continue
       if sect in ("[tables]","[views]"):
           m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
           if m: names.add(m.group(1).lower())
   if not names: die("No table/view names found in schema.txt.")
   return names
ALLOWED = allowed_from_schema(SCHEMA)


# ---- DB connect ----
conn = cursor = None
try:
   import pymysql
   conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
                          database=DB_NAME, cursorclass=pymysql.cursors.DictCursor, autocommit=True)
   cursor = conn.cursor()
except Exception:
   try:
       import mysql.connector
       conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME)
       cursor = conn.cursor(dictionary=True)
   except Exception as e:
       die(f"could not connect to MySQL: {e}")


# ---- Ollama ----
def ollama_generate(prompt: str) -> str:
   payload = {"model": MODEL, "prompt": prompt, "stream": False}
   data = json.dumps(payload).encode("utf-8")
   req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
   with urllib.request.urlopen(req, timeout=120) as resp:
       obj = json.loads(resp.read().decode("utf-8"))
   return (obj.get("response") or "").strip()


def parse_llm_sql(text: str) -> str:
   t = text.strip()
   # strip backtick fences
   if t.startswith("```"):
       t = re.sub(r"^```[a-zA-Z]*\n", "", t).rstrip()
       if t.endswith("```"):
           t = t[:-3]
   # try direct JSON
   try:
       obj = json.loads(t)
       sql = (obj.get("sql") or "").strip()
       if sql:
           return sql
   except Exception:
       pass
   # try JSON object embedded in text (collapse newlines to spaces)
   m = re.search(r"\{.*\}", t, flags=re.DOTALL)
   if m:
       candidate = re.sub(r"\s*\n\s*", " ", m.group(0))
       try:
           obj = json.loads(candidate)
           sql = (obj.get("sql") or "").strip()
           if sql:
               return sql
       except Exception:
           pass
   # last resort: pull first SELECT ... ; block
   m = re.search(r"(?is)\bselect\b.*?;", t)
   if m:
       return m.group(0).strip()
   raise ValueError("LLM did not return parseable SQL.")








# ---- detect raw SQL ----
SQL_START = re.compile(r"(?is)^\s*(select|with)\b")
SQL_PATTERN = re.compile(r"(?is)\bselect\b.+\bfrom\b")
def looks_like_sql(s: str) -> bool:
   s = s.strip()
   return bool(SQL_START.match(s) or SQL_PATTERN.search(s) or re.search(r"(?is)\bcount\s*\(.*\)\s+from\b", s))


# ---- extract Top-N from question or from SQL LIMIT ----
TOPN_Q_RE = re.compile(r"(?i)\btop\s*([0-9]{1,3})\b")
LIMIT_SQL_RE = re.compile(r"(?i)\blimit\s+(\d+)\b")
def extract_topn_from_question(q: str):
   m = TOPN_Q_RE.search(q or "");
   if not m: return None
   n = max(1, min(int(m.group(1)), MAX_LIMIT))
   return n
def extract_limit_from_sql(sql: str):
   m = LIMIT_SQL_RE.search(sql or "")
   return int(m.group(1)) if m else None


# ---- SQL safety ----
def one_statement_only(sql: str) -> bool:
   s = sql.strip()
   return ";" not in s[:-1]  # optional trailing ; only


def enforce_limit(sql: str, requested_n: int | None) -> str:
   if re.search(r"(?i)\bcount\s*\(", sql): return sql  # pure counts: no limit needed
   target = requested_n if requested_n is not None else extract_limit_from_sql(sql) or MAX_LIMIT
   target = min(target, MAX_LIMIT)
   if re.search(r"(?i)\blimit\b", sql):
       return re.sub(r"(?i)\blimit\s+(\d+)", lambda m: f"LIMIT {target}", sql)
   return sql.rstrip(";") + f" LIMIT {target}"


def referenced_tables(sql: str):
   f = re.findall(r"(?i)\bfrom\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", sql)
   j = re.findall(r"(?i)\bjoin\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", sql)
   return {t.lower() for t in (f + j)}


def validate_sql(sql: str, requested_n: int | None) -> str:
   s = sql.strip()
   if not re.match(r"(?is)^\s*select\b", s): raise ValueError("Only SELECT is allowed.")
   if not one_statement_only(s):            raise ValueError("Multiple statements are not allowed.")
   for kw in DISALLOWED:
       if re.search(rf"(?i)\b{kw}\b", s):   raise ValueError(f"Disallowed keyword: {kw}")
   miss = [t for t in referenced_tables(s) if t not in ALLOWED]
   if miss: raise ValueError(f"Unknown table/view: {', '.join(miss)}")
   s = enforce_limit(s, requested_n)
   return s


# ---- NL→SQL prompt ----
SYS_RULES = f"""
Convert English to ONE MySQL SELECT for Gravity Books.
- Use ONLY schema tables/views; SELECT-only; ONE statement.
- Always QUALIFY columns with table aliases when joins are used (e.g., b.title AS book_title).
- When using base tables, the per-line amount is order_line.price (not line_total).
- If using the view v_order_items, you may use line_total and title directly.
- For 'top N ... in <year>', filter by YEAR(order_date)=<year> (orders are in 2020–2023).
- For date windows, use order_date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.
- Never treat deictic words like here/this/these/there/it as literal filters.
- If not 'top N', include LIMIT <= {MAX_LIMIT} unless a pure COUNT.
Respond ONLY as compact one-line JSON with no newlines inside strings: {{"sql":"..."}}
"""








FEWSHOT = """
Q: HOW MANY BOOKS?
A: {"sql":"SELECT COUNT(*) AS book_count FROM book;"}


Q: top 5 authors with most sales
A: {"sql":"SELECT a.author_name, SUM(ol.price) AS sales FROM order_line ol JOIN book b ON ol.book_id=b.book_id JOIN book_author ba ON b.book_id=ba.book_id JOIN author a ON a.author_id=ba.author_id GROUP BY a.author_name ORDER BY sales DESC LIMIT 5;"}


Q: top 10 publishers by revenue in 2023
A: {"sql":"SELECT p.publisher_name, SUM(ol.price) AS revenue FROM order_line ol JOIN book b ON ol.book_id=b.book_id JOIN publisher p ON b.publisher_id=p.publisher_id JOIN cust_order co ON co.order_id = ol.order_id WHERE YEAR(co.order_date)=2023 GROUP BY p.publisher_name ORDER BY revenue DESC LIMIT 10;"}


Q: items for order 1234 (prefer the view)
A: {"sql":"SELECT voi.title, voi.line_total FROM v_order_items voi WHERE voi.order_id=1234 LIMIT 200;"}


Q: items for order 1234 (no view)
A: {"sql":"SELECT b.title AS book_title, ol.price AS line_total FROM order_line ol JOIN book b ON b.book_id=ol.book_id WHERE ol.order_id=1234 LIMIT 200;"}


Q: orders between 2021-01-01 and 2021-12-31
A: {"sql":"SELECT order_id, order_date FROM cust_order WHERE order_date BETWEEN '2021-01-01' AND '2021-12-31' ORDER BY order_date LIMIT 200;"}
"""












def build_sql_prompt(question: str) -> str:
   return f"{SYS_RULES}\n[Schema]\n{SCHEMA}\n\n{FEWSHOT}\nQ: {question}\nA: "


# ---- execute ----
def run_sql(sql: str):
   s = sql.strip().rstrip(";")
   cursor.execute(s)
   try:
       return cursor.fetchall()
   except Exception:
       r = cursor.fetchone()
       return [r] if r else []


# ---- result analysis & rendering ----
NAME_HINTS = {"publisher_name","author_name","customer_name","language_name","title","name"}
def analyze_rows(rows):
   if not rows: return {"type":"empty"}
   if len(rows)==1 and isinstance(rows[0], dict) and len(rows[0])==1:
       k,v = next(iter(rows[0].items())); return {"type":"count","key":k,"value":v}
   keys = list(rows[0].keys())
   label = next((k for k in keys if k in NAME_HINTS or k.endswith("_name")), keys[0])
   # pick up to two numeric metrics (by type or name patterns)
   num_keys = [k for k in keys if k != label and isinstance(rows[0].get(k),(int,float))]
   if not num_keys:
       num_keys = [k for k in keys if k != label and re.search(r"(revenue|sales|units|total|count|amount|price|qty|n)", k, re.I)]
   return {"type":"list","label":label,"metrics":num_keys[:2],"keys":keys}


def llm_one_liner(question: str, value_key: str, value):
   # Use LLM only for counts/simple sentences (fast)
   system = ("Write ONE short sentence as the final answer. "
             "Do not reveal SQL or internals. Be factual and concise.")
   meta = {"question": question, "metric": value_key, "value": value}
   prompt = f"{system}\nContext JSON:\n{json.dumps(meta, ensure_ascii=False)}\nAnswer:"
   try:
       return ollama_generate(prompt).strip()
   except Exception:
       # deterministic fallback
       return f"{value_key.replace('_',' ').title()}: {value}."


def bullets(rows, label_key, metrics, n_show):
   lines=[]
   for r in rows[:n_show]:
       if metrics:
           vals = "; ".join(f"{m}={r.get(m,'')}" for m in metrics)
           lines.append(f"- {r.get(label_key,'')} — {vals}")
       else:
           lines.append(f"- {r.get(label_key,'')}")
   return "\n".join(lines)


def compact_table(rows, n_show=None):
   if not rows: return ""
   keys = list(rows[0].keys())
   show = rows if n_show is None else rows[:n_show]
   widths = [max(len(str(k)), max(len(str(r.get(k,''))) for r in show)) for k in keys]
   head = " | ".join(k.ljust(w) for k,w in zip(keys,widths))
   sep  = "-+-".join("-"*w for w in widths)
   body = "\n".join(" | ".join(str(r.get(k,"")).ljust(w) for k,w in zip(keys,widths)) for r in show)
   return f"{head}\n{sep}\n{body}"


# ---- audit ----
def audit(event: dict):
   event = {"ts": now_iso(), **event}
   with open(AUDIT_FILE, "a", encoding="utf-8") as f:
       f.write(json.dumps(event, ensure_ascii=False) + "\n")


# --- REPL ---
print(f"Lab06 — Gravity Books Chatbot (model {MODEL})")
print("Type /help for commands. Ask in English or paste SQL.")


while True:
   try:
       user = input("\n> ").strip()
   except (EOFError, KeyboardInterrupt):
       print("\nGoodbye."); break
   if not user:
       continue


   # Handle slash-commands FIRST — never send them to LLM/DB
   cmd = parse_slash_command(user)
   if cmd is not None:  # user typed something starting with '/'
       if cmd in {"exit", "quit", "q"}:
           print("Goodbye.")
           break
       elif cmd in {"help", "h", "?"}:
           print(HELP_TEXT)
           continue
       else:
           print(f"Unknown command '/{cmd}'. Type /help.")
           continue


   # Also support plain 'help' or 'exit' (without slash) just in case
   lo = user.lower()
   if lo in {"exit", "quit"}:
       print("Goodbye.")
       break
   if lo in {"help", "h", "?"}:
       print(HELP_TEXT)
       continue




   # 1) decide requested N
   requested_n_q = None if looks_like_sql(user) else extract_topn_from_question(user)


   # 2) NL→SQL or raw SQL
   try:
       if looks_like_sql(user):
           sql = user
           src = "sql"
       else:
           sql_raw = parse_llm_sql(ollama_generate(build_sql_prompt(user)))
           sql = validate_sql(sql_raw, requested_n_q)
           if not sql: raise ValueError("Empty SQL from model.")
           src = "nl"
       # 3) validate + enforce LIMIT (from question or SQL)
       sql = validate_sql(sql, requested_n_q)
   except Exception as e:
       print(f"Sorry, I couldn’t form a safe query. ({e})")
       audit({"q": user, "stage":"validate", "ok":False, "error":str(e)})
       continue


   # 4) execute
   try:
       rows = run_sql(sql)
   except Exception as e:
       msg = str(e)
       # auto-fix common mistake once: 'line_total' on base table
       if "Unknown column" in msg and "line_total" in msg and "order_line" in sql:
           sql_alt = sql.replace("line_total", "price")
           rows = run_sql(sql_alt)
           sql = sql_alt
       else:
           raise








   # 5) render (LLM only for count)
   shape = analyze_rows(rows)
   limit_sql = extract_limit_from_sql(sql)
   requested_n = requested_n_q or limit_sql
   n_returned = len(rows)


   if shape["type"] == "empty":
       print("No matches found.")
       mode = "none"
   elif shape["type"] == "count":
       one_line = llm_one_liner(user, shape["key"], shape["value"])
       print(one_line)
       mode = "count"
   else:
       label, metrics = shape["label"], shape["metrics"]
       # If user asked top N, but DB has fewer rows, acknowledge once.
       prefix = ""
       if requested_n is not None and n_returned < requested_n:
           prefix = f"(Only {n_returned} found for requested top {requested_n}.)\n"
       if n_returned <= SMALL_BULLET_THRESHOLD:
           print(prefix + bullets(rows, label, metrics, n_returned))
           mode = "bullets"
       else:
           # big lists => fast table; show up to LIMIT rows
           print(prefix + f"Showing {n_returned} row(s).")
           print(compact_table(rows))
           mode = "table"


   audit({"q": user, "stage":"done", "ok":True, "src":src, "sql":sql,
          "rows":n_returned, "mode":mode, "req_n":requested_n})







