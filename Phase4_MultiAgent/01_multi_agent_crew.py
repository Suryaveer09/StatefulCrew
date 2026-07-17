# ─────────────────────────────────────────────────────────────────
# Phase 4: Multi-Agent Crew — a Supervisor coordinating three
# specialist agents (SQL, Analysis, Report) via LangGraph, using the
# supervisor pattern: every specialist hands control back to the
# Supervisor rather than answering the user directly.
# ─────────────────────────────────────────────────────────────────

from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages

from langchain_deepseek import ChatDeepSeek

import sqlite3
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()  # pulls DEEPSEEK_API_KEY and LANGSMITH_* vars from .env

DB_PATH = "../Phase2_Tools/chinook.db"


# State shared across every node in the crew. "next" is set by the
# Supervisor each turn to tell the graph which node runs next.
class CrewState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


MEMBERS = ["sql_agent", "analysis_agent", "report_agent"]

# One shared model instance used by every node below
llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)


# ─────────────────────────────────────────────────────────────────
# Supervisor — routes via a tool call, with a SANITIZED view of history
# ─────────────────────────────────────────────────────────────────
# Bug found: passing raw state["messages"] (containing other agents' real
# tool_calls, e.g. sql_agent's run_sql_query calls) to the supervisor caused
# DeepSeek to imitate that tool name instead of using its own bound `route`
# tool. Fix: strip raw tool-call structures into plain text summaries
# before the supervisor ever sees them.

# route() is the mechanism the Supervisor uses to decide what happens
# next. Routing is done as a tool call rather than structured JSON output
# (with_structured_output + json_mode) because json_mode proved unreliable
# for this specific node — tool-calling has been the one reliable
# mechanism since Phase 2's SQL agent.
@tool
def route(next_agent: Literal["sql_agent", "analysis_agent", "report_agent", "FINISH"]) -> str:
    """Call this to route to the next agent in the crew, or FINISH when the task is complete."""
    return next_agent


# Only the route tool is bound here — the supervisor has no access to
# run_sql_query or any other specialist's tools, by design.
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
    can't imitate other agents' literal tool_calls JSON.
    """
    context = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # User turns pass through unchanged
            context.append(msg)
        elif isinstance(msg, ToolMessage):
            # Raw tool output becomes a plain-text note instead of a
            # "tool" role message — the supervisor sees the DATA, not
            # the fact that a specific tool call produced it
            context.append(HumanMessage(content=f"[Tool result]: {msg.content[:300]}"))
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                # This is the critical line: instead of exposing the
                # literal tool_calls structure (which the supervisor's
                # own model previously imitated), we describe it in text
                tool_names = ", ".join(c["name"] for c in msg.tool_calls)
                context.append(HumanMessage(content=f"[{msg.name or 'agent'} used a tool: {tool_names}]"))
            elif msg.content:
                context.append(HumanMessage(content=f"[{msg.name or 'agent'} said]: {msg.content}"))
    return context


def supervisor_node(state: CrewState) -> dict:
    clean_context = build_supervisor_context(state["messages"])
    ai_msg = router_llm.invoke([SUPERVISOR_SYSTEM_PROMPT] + clean_context)

    if ai_msg.tool_calls:
        call_args = ai_msg.tool_calls[0]["args"]
        next_agent = call_args.get("next_agent")
        if next_agent is None:
            # Defensive fallback: if the model calls route() with an
            # unexpected argument shape, log it and finish cleanly
            # instead of crashing on a KeyError.
            print("route() was called with unexpected args:", call_args)
            print("Full tool call:", ai_msg.tool_calls[0])
            next_agent = "FINISH"
    else:
        # The model answered in plain text instead of calling route() —
        # treat that as "I think we're done" rather than erroring out.
        next_agent = "FINISH"
        print("Supervisor didn't call route() — defaulting to FINISH. Response was:", ai_msg.content)

    return {"next": next_agent}


# ─────────────────────────────────────────────────────────────────
# SQL Agent — now grounded with the real Chinook schema
# ─────────────────────────────────────────────────────────────────
# Bug found: with no schema info, the model guessed a plausible-but-wrong
# table name ("invoice_items" instead of the real "InvoiceLine"). Fix:
# give it the actual schema up front instead of making it guess.
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
    # Basic safety guard — only SELECT statements are allowed through
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
        # Cap output size so a huge result set doesn't flood the model's context
        return ", ".join(columns) + "\n" + "\n".join(str(r) for r in rows[:20])
    except Exception as e:
        # Return SQL errors as text (not a crash) so the model can see
        # what went wrong and potentially write a corrected query
        conn.close()
        return f"SQL Error: {e}"


sql_llm = llm.bind_tools([run_sql_query])


def sql_agent_node(state: CrewState) -> dict:
    # Note: this node still sees the FULL raw state["messages"], unlike
    # the supervisor's sanitized context — safe here because sql_agent
    # only ever runs once per supervisor cycle and needs to see real
    # tool results to reason about follow-up queries.
    ai_msg = sql_llm.invoke([SQL_AGENT_SYSTEM_PROMPT] + state["messages"])
    new_messages = [ai_msg]
    for call in ai_msg.tool_calls:
        result = run_sql_query.invoke(call["args"])
        new_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": new_messages}


def analysis_agent_node(state: CrewState) -> dict:
    # Not bound to any tools — this node only ever reasons over text
    # already in the conversation, it never queries the database itself
    analysis_prompt = (
        "You are a data analyst. Review the SQL results in the conversation above. "
        "Identify any notable patterns, totals, or anomalies worth mentioning. Be brief."
    )
    response = llm.invoke(state["messages"] + [HumanMessage(analysis_prompt)])
    # name="analysis_agent" tags this message so build_supervisor_context
    # (and any future readers of state) can tell which agent said it
    return {"messages": [AIMessage(content=response.content, name="analysis_agent")]}


def report_agent_node(state: CrewState) -> dict:
    report_prompt = (
        "Write a clear, final answer to the user's original question, using the data "
        "and analysis gathered above. Be concise and cite the actual numbers."
    )
    response = llm.invoke(state["messages"] + [HumanMessage(report_prompt)])
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

# The supervisor's "next" field drives routing — this conditional edge
# is the crew's dispatcher, sending control to whichever node the
# supervisor decided on (or straight to END on "FINISH")
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"sql_agent": "sql_agent", "analysis_agent": "analysis_agent",
     "report_agent": "report_agent", "FINISH": END},
)

# Every specialist reports back to the supervisor when done — this is
# what makes the supervisor pattern a loop rather than a straight line
builder.add_edge("sql_agent", "supervisor")
builder.add_edge("analysis_agent", "supervisor")
builder.add_edge("report_agent", "supervisor")

# MemorySaver gives the graph checkpointed memory, keyed by thread_id —
# lets a follow-up question in the same session use prior context
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "crew-session-1"}}
result = graph.invoke(
    {"messages": [HumanMessage(
        "What were the top 5 genres by total sales, and is there anything unusual about the numbers?"
    )]},
    config=config,
)
print(result["messages"][-1].content)