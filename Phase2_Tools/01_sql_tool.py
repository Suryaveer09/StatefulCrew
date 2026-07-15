# ─────────────────────────────────────────────────────────────────
# Phase 2: Tools — teaching an LLM to query a real SQLite database
#
# This script demonstrates the ReAct loop: the model reasons about a
# question, requests a tool call, reads the result, and repeats until
# it has enough information to give a final natural-language answer.
# ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()  # pulls DEEPSEEK_API_KEY and LANGSMITH_* vars from .env into the environment

import sqlite3
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, ToolMessage

DB_PATH = "chinook.db"  # the sample music-store database sitting in this folder


# ─────────────────────────────────────────────────────────────────
# The tool
# ─────────────────────────────────────────────────────────────────
# @tool turns this plain Python function into something the LLM can "see"
# and choose to call. LangChain reads the function's docstring and type
# hints to build a schema the model uses to decide when/how to call it —
# so the docstring below isn't just documentation, it's part of the
# model's instructions.
@tool
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against the Chinook music store database
    and return the results. Use standard SQLite syntax. Only SELECT statements
    are allowed — no INSERT, UPDATE, DELETE, or DROP.
    """
    # Safety guard: block anything that isn't a SELECT before it touches
    # the database. This is a simple string check, not real SQL parsing —
    # good enough for a learning project, not production-grade.
    if not query.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed."

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(query)

        # cursor.description holds column metadata; we only need the names
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query ran successfully but returned no rows."

        # Tools must return plain text — the model reads strings, not
        # Python objects — so we format the rows into a simple text table.
        result = ", ".join(columns) + "\n"
        result += "\n".join(str(row) for row in rows[:20])  # cap size so we don't flood the model's context
        return result

    except Exception as e:
        # If the model writes bad SQL (wrong column, typo, missing join),
        # we don't want the script to crash — return the error as text so
        # the model can see what went wrong and try a corrected query on
        # its next turn through the loop below.
        conn.close()
        return f"SQL Error: {e}"


# ─────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────
llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)

# bind_tools() doesn't let the model run the tool — it only makes the model
# aware the tool exists, along with its name, description, and expected
# arguments, so it can request the tool when relevant.
llm_with_tools = llm.bind_tools([run_sql_query])


# ─────────────────────────────────────────────────────────────────
# The ReAct loop: reason → act → observe → repeat until done
# ─────────────────────────────────────────────────────────────────
# Harder questions (like this one, which needs a join across Customer and
# Invoice) often can't be answered in a single tool call. The model may
# need to explore the schema, try a query, see the result, and refine —
# an unknown number of rounds. This loop keeps going until the model
# responds with no further tool calls, meaning it's ready to give a
# real answer.

question = "Which customer has spent the most money in total, and how much did they spend?"

# messages is the running conversation history, shared across every round
# of the loop so the model always has full context of what it already tried.
messages = [HumanMessage(question)]

while True:
    # Ask the model what it wants to do next, given everything so far.
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # No tool calls in the response means the model is done reasoning and
    # has produced its final natural-language answer — exit the loop.
    if not ai_msg.tool_calls:
        break

    # Otherwise, run every requested tool call and feed each result back
    # into the conversation as a ToolMessage. tool_call_id links each
    # result to its specific request, which matters if the model ever
    # requests multiple tool calls in a single turn.
    for call in ai_msg.tool_calls:
        result = run_sql_query.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # Loop back: the model sees the new tool result and decides whether
    # it has enough information now, or needs to call the tool again.

print("\nFinal answer:\n", ai_msg.content)