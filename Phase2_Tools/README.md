# Phase 2 — Tools

Goal: give the model something to *do*, not just something to say — a real tool it can call to query a SQLite database, and see how far tool-calling reasoning goes on its own.

## What's in this folder

| File | What it demonstrates |
|---|---|
| `sql_tool.py` | A custom `@tool`-decorated SQL function, bound to DeepSeek, driven through a full ReAct loop |
| `chinook.db` | Sample music-store database (artists, albums, tracks, customers, invoices) — gitignored, not committed |

## Key concepts

**A tool is just a Python function with a schema attached.** `@tool` reads the function's docstring and type hints to tell the model what the tool does and what arguments it expects. The docstring isn't just documentation — it's effectively part of the prompt:

```python
@tool
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against the Chinook music store database..."""
```

**`bind_tools()` doesn't let the model run anything.** It only makes the model *aware* a tool exists. The model can request a tool call with specific arguments, but the calling code is always responsible for actually executing it and returning the result as text.

**The ReAct loop** — reason, act, observe, repeat — is the pattern underneath almost all tool-using agents:
```python
while True:
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        break  # model has enough info to answer

    for call in ai_msg.tool_calls:
        result = run_sql_query.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
```
Simple questions resolve in one round. Harder questions — especially ones needing a join across tables the model has never seen the schema for — can take several rounds of "try a query, read the result, try again."

## What broke (and why it mattered)

**Bug 1 — silent no-op with a single tool-call round.** My first version of the script called the model once, ran the tool, fed the result back, and printed `final.content` — assuming one round of tool-calling was always enough. For a simple query ("first 5 artists alphabetically") this worked fine. For a harder question requiring a join:

```python
question = "Which customer has spent the most money in total, and how much did they spend?"
```

...the script printed a blank final answer. The model had requested *another* tool call on the second round — probably to explore the schema or refine its join — but the script only handled one round and never checked for a second `tool_calls` list. It silently discarded the model's follow-up request and tried to print `.content` on a response that had none.

**Fix:** replace the fixed two-step call with a `while True` loop that keeps going until the model responds with **no further tool calls** — meaning it's actually ready to answer, regardless of whether that takes 1 round or 5.

**Result after the fix:**
```
Final answer:
 The customer who has spent the most money is Helena Holý (email: hholy@gmail.com),
 with a total spend of $49.62.
```
Correct — and the model got there without me telling it anything about the `Customer` or `Invoice` table structure. It figured out the join, and the correct aggregation, entirely from tool-calling reasoning over multiple rounds.

**Takeaway:** never assume tool-calling resolves in a fixed number of rounds. Any real agent needs a loop (or, as of Phase 3, a graph) that keeps going until the model signals it's actually done — not until *you* stop asking.

## How to run

```bash
# one-time setup — download the sample database
curl -L -o chinook.db https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite

python sql_tool.py
```

## Why this matters for Phase 3+

This `while True` loop is exactly what LangGraph replaces with a structured, inspectable graph. Seeing the manual version first — and hitting a real bug in it — makes it obvious *why* LangGraph's abstraction exists, instead of it just being "a new syntax to memorize."