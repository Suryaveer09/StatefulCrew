from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages

from langchain_deepseek import ChatDeepSeek

import sqlite3
import time
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

DB_PATH = "../Phase2_Tools/chinook.db"


class CrewState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    iterations: int  # tracks how many times the supervisor has run — guards against infinite loops


MEMBERS = ["sql_agent", "analysis_agent", "report_agent"]
MAX_ITERATIONS = 6  # soft cap — supervisor + 3 specialists should rarely need more than a couple rounds each

llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)


# ─────────────────────────────────────────────────────────────────
# Generic retry wrapper — protects any node's LLM call against
# transient API errors (network blips, rate limits), not meant to
# paper over deterministic bugs, just real-world flakiness.
# ─────────────────────────────────────────────────────────────────
def invoke_with_retry(chain_or_llm, inputs, max_attempts: int = 3, node_name: str = "node"):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chain_or_llm.invoke(inputs)
        except Exception as e:
            last_error = e
            print(f"[{node_name}] call failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(2 ** (attempt - 1))  # exponential backoff: 1s, 2s, 4s
    raise last_error


# ─────────────────────────────────────────────────────────────────
# Supervisor — routes via a tool call, with a sanitized view of history
# and a hard iteration cap to prevent infinite loops.
# ─────────────────────────────────────────────────────────────────
@tool
def route(next_agent: Literal["sql_agent", "analysis_agent", "report_agent", "FINISH"]) -> str:
    """Call this to route to the next agent in the crew, or FINISH when the task is complete."""
    return next_agent


router_llm = llm.bind_tools([route])

SUPERVISOR_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a supervisor coordinating a data analysis crew: sql_agent, analysis_agent, report_agent.\n"
    "You MUST call the `route` tool on every turn to say who acts next — never answer in plain text.\n"
    "- sql_agent: runs SQL queries against the Chinook database to fetch raw data\n"
    "- analysis_agent: examines query results for patterns, totals, or anomalies\n"
    "- report_agent: writes the final natural-language answer for the user\n"
    "Route to sql_agent first if no data has been fetched yet. Route to report_agent only "
    "once you have everything needed for a complete answer. Call route with FINISH once "
    "report_agent has already produced the final answer."
))


def build_supervisor_context(messages: list) -> list:
    """Give the supervisor a clean, tool-call-free view of progress so it
    can't imitate other agents' literal tool_calls JSON (Phase 4 bug).
    """
    context = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            context.append(msg)
        elif isinstance(msg, ToolMessage):
            context.append(HumanMessage(content=f"[Tool result]: {msg.content[:300]}"))
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_names = ", ".join(c["name"] for c in msg.tool_calls)
                context.append(HumanMessage(content=f"[{msg.name or 'agent'} used a tool: {tool_names}]"))
            elif msg.content:
                context.append(HumanMessage(content=f"[{msg.name or 'agent'} said]: {msg.content}"))
    return context


def supervisor_node(state: CrewState) -> dict:
    iterations = state.get("iterations", 0) + 1

    # Hard guardrail: force a clean finish rather than let the graph spin
    # forever if the supervisor keeps deciding "not done yet."
    if iterations > MAX_ITERATIONS:
        print(f"Hit max iterations ({MAX_ITERATIONS}) — forcing FINISH to avoid an infinite loop.")
        return {"next": "FINISH", "iterations": iterations}

    clean_context = build_supervisor_context(state["messages"])
    ai_msg = invoke_with_retry(
        router_llm, [SUPERVISOR_SYSTEM_PROMPT] + clean_context, node_name="supervisor"
    )

    if ai_msg.tool_calls:
        call_args = ai_msg.tool_calls[0]["args"]
        next_agent = call_args.get("next_agent")
        if next_agent is None:
            print("route() was called with unexpected args:", call_args)
            print("Full tool call:", ai_msg.tool_calls[0])
            next_agent = "FINISH"
    else:
        next_agent = "FINISH"
        print("Supervisor didn't call route() — defaulting to FINISH. Response was:", ai_msg.content)

    return {"next": next_agent, "iterations": iterations}


# ─────────────────────────────────────────────────────────────────
# SQL Agent — grounded with the real Chinook schema
# ─────────────────────────────────────────────────────────────────
SQL_AGENT_SYSTEM_PROMPT = SystemMessage(content=(
    "You write SQLite SELECT queries against the Chinook database. The real schema is:\n"
    "Artist(ArtistId, Name)\n"
    "Album(AlbumId, Title, ArtistId)\n"
    "Track(TrackId, Name, AlbumId, GenreId, MediaTypeId, Composer, Milliseconds, Bytes, UnitPrice)\n"
    "Genre(GenreId, Name)\n"
    "Customer(CustomerId, FirstName, LastName, Email, City, Country, ...)\n"
    "Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, Total)\n"
    "InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)\n"
    "Use exactly these table and column names — do not guess or invent names like "
    "'invoice_items'. For sales by genre, join Track -> InvoiceLine -> Genre."
))


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


sql_llm = llm.bind_tools([run_sql_query])


def sql_agent_node(state: CrewState) -> dict:
    ai_msg = invoke_with_retry(
        sql_llm, [SQL_AGENT_SYSTEM_PROMPT] + state["messages"], node_name="sql_agent"
    )
    new_messages = [ai_msg]
    for call in ai_msg.tool_calls:
        result = run_sql_query.invoke(call["args"])
        new_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": new_messages}


def analysis_agent_node(state: CrewState) -> dict:
    analysis_prompt = (
        "You are a data analyst. Review the SQL results in the conversation above. "
        "Identify any notable patterns, totals, or anomalies worth mentioning. Be brief."
    )
    response = invoke_with_retry(
        llm, state["messages"] + [HumanMessage(analysis_prompt)], node_name="analysis_agent"
    )
    return {"messages": [AIMessage(content=response.content, name="analysis_agent")]}


def report_agent_node(state: CrewState) -> dict:
    report_prompt = (
        "Write a clear, final answer to the user's original question, using the data "
        "and analysis gathered above. Be concise and cite the actual numbers."
    )
    response = invoke_with_retry(
        llm, state["messages"] + [HumanMessage(report_prompt)], node_name="report_agent"
    )
    return {"messages": [AIMessage(content=response.content, name="report_agent")]}


# ─────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────
builder = StateGraph(CrewState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("sql_agent", sql_agent_node)
builder.add_node("analysis_agent", analysis_agent_node)
builder.add_node("report_agent", report_agent_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"sql_agent": "sql_agent", "analysis_agent": "analysis_agent",
     "report_agent": "report_agent", "FINISH": END},
)

builder.add_edge("sql_agent", "supervisor")
builder.add_edge("analysis_agent", "supervisor")
builder.add_edge("report_agent", "supervisor")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# recursion_limit is LangGraph's own hard backstop — a second, independent
# layer of defense in case our own MAX_ITERATIONS guardrail has a bug.
# If the graph exceeds this many steps, LangGraph raises GraphRecursionError.
config = {
    "configurable": {"thread_id": "crew-session-1"},
    "recursion_limit": 15,
}

result = graph.invoke(
    {"messages": [HumanMessage(
        "What were the top 5 genres by total sales, and is there anything unusual about the numbers?"
    )],
     "iterations": 0},
    config=config,
)
print(result["messages"][-1].content)
print(f"\n(Completed in {result['iterations']} supervisor iterations)")
