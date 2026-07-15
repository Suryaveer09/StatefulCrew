# ─────────────────────────────────────────────────────────────────
# Phase 3: LangGraph Basics — rebuilding the Phase 2 SQL agent as an
# actual graph, with persistent conversation memory via a checkpointer.
# ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()  # pulls DEEPSEEK_API_KEY and LANGSMITH_* vars from .env

import sqlite3
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Reuse the same Chinook database from Phase 2 rather than duplicating it
DB_PATH = "../Phase2_Tools/chinook.db"


# ─────────────────────────────────────────────────────────────────
# The tool (identical to Phase 2 — the tool itself doesn't change,
# only how it's wired into the agent's control flow)
# ─────────────────────────────────────────────────────────────────
@tool
def run_sql_query(query: str) -> str:
    """Execute a read-only SELECT query against the Chinook database."""
    if not query.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed."

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(query)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No rows returned."
        return ", ".join(columns) + "\n" + "\n".join(str(r) for r in rows[:20])
    except Exception as e:
        conn.close()
        return f"SQL Error: {e}"


# ─────────────────────────────────────────────────────────────────
# State — the shared data structure every node in the graph reads
# from and writes to.
# ─────────────────────────────────────────────────────────────────
# Annotated[list, add_messages] is a "reducer": it tells LangGraph HOW
# to merge a node's returned messages into the existing state, rather
# than just overwriting it. add_messages specifically APPENDS new
# messages to the list — which is exactly what a conversation history
# needs. Without this, every node's return value would replace the
# whole messages list instead of adding to it.
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────────
# Model + tool binding (same as Phase 2)
# ─────────────────────────────────────────────────────────────────
llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)
llm_with_tools = llm.bind_tools([run_sql_query])


def agent_node(state: State) -> dict:
    """The 'thinking' node: looks at the full conversation so far and
    decides whether to answer directly or request a tool call. Returns
    only the NEW message — the add_messages reducer takes care of
    appending it to the existing history.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ─────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("agent", agent_node)
# ToolNode is a prebuilt LangGraph node: it inspects the last AI message
# for tool_calls, runs each one, and returns the results as ToolMessages —
# this replaces the manual "for call in ai_msg.tool_calls: ..." loop from
# Phase 2.
builder.add_node("tools", ToolNode([run_sql_query]))

builder.add_edge(START, "agent")  # every run starts at the agent node

# tools_condition is a prebuilt router: it checks the last message for
# pending tool calls. If there are any, it routes to "tools"; if not,
# it routes to END. This one line replaces the entire
# `if not ai_msg.tool_calls: break` check from the Phase 2 while loop.
builder.add_conditional_edges("agent", tools_condition)

# After the tools run, go back to the agent so it can read the results
# and decide what to do next — call another tool, or give a final answer.
# This edge is what makes the loop actually loop.
builder.add_edge("tools", "agent")

# ─────────────────────────────────────────────────────────────────
# Checkpointing — gives the graph persistent memory across calls
# ─────────────────────────────────────────────────────────────────
# MemorySaver stores conversation state in RAM, keyed by thread_id.
# It's lost when the process exits — fine for learning/dev, not for
# production (that's what PostgresSaver etc. are for later).
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# thread_id identifies a single ongoing conversation. Reusing the same
# thread_id across multiple .invoke() calls lets the graph "remember"
# everything said before, without you manually re-sending prior messages.
config = {"configurable": {"thread_id": "session-1"}}


# ─────────────────────────────────────────────────────────────────
# First turn — a question with no prior context
# ─────────────────────────────────────────────────────────────────
result = graph.invoke(
    {"messages": [HumanMessage(
        "Which customer has spent the most money in total, and how much did they spend?"
    )]},
    config=config,
)
print(result["messages"][-1].content)


# ─────────────────────────────────────────────────────────────────
# Second turn — a follow-up that depends on memory of the first turn.
# "they" only resolves correctly because the checkpointer (same
# thread_id) is feeding the full prior conversation back into state
# automatically — this call only sends the NEW human message.
# ─────────────────────────────────────────────────────────────────
follow_up = graph.invoke(
    {"messages": [HumanMessage("What city are they from?")]},
    config=config,
)
print(follow_up["messages"][-1].content)
