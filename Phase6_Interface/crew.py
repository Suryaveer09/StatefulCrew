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

DB_PATH = "../Phase2_Tools/chinook.db"  # Phase 6 folder structure — sibling to Phase2_Tools


class CrewState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    iterations: int


MEMBERS = ["sql_agent", "analysis_agent", "report_agent"]
MAX_ITERATIONS = 6

llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)


def invoke_with_retry(chain_or_llm, inputs, max_attempts: int = 3, node_name: str = "node"):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chain_or_llm.invoke(inputs)
        except Exception as e:
            last_error = e
            print(f"[{node_name}] call failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(2 ** (attempt - 1))
    raise last_error


def get_latest_user_question(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ─────────────────────────────────────────────────────────────────
# GENERALIZED sanitizer — was previously only used for the supervisor
# (build_supervisor_context). Renamed and now applied to EVERY node's
# model call, not just the supervisor's. Root cause of a real regression
# found via Streamlit multi-turn testing: with enough turns accumulated
# in a long session, ANY node seeing raw tool_calls/ToolMessage objects
# from earlier questions could imitate or leak that structure — not just
# the supervisor. analysis_agent and report_agent aren't bound to any
# tools, so when they tried to imitate one, it leaked out as literal
# text in the final answer instead of being executed.
# ─────────────────────────────────────────────────────────────────
def build_sanitized_context(messages: list) -> list:
    """Strip raw tool_calls/ToolMessage structures out of what any node's
    model call sees, replacing them with plain-text summaries. Preserves
    full session history (so follow-up questions still work) without ever
    exposing raw tool-call JSON to a node that shouldn't act on it.
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


# ─────────────────────────────────────────────────────────────────
# Supervisor
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
    "Route to sql_agent first if no data has been fetched yet for the CURRENT question. "
    "Route to report_agent only once you have everything needed for a complete answer. "
    "Call route with FINISH once report_agent has already produced the final answer.\n"
    "If sql_agent explains that the schema doesn't contain the data needed to answer "
    "the question, do NOT route back to sql_agent hoping for a different result — "
    "route directly to report_agent so it can explain this limitation to the user."
))


def supervisor_node(state: CrewState) -> dict:
    iterations = state.get("iterations", 0) + 1

    if iterations > MAX_ITERATIONS:
        # Never let FINISH skip report_agent — that's what caused raw tool
        # dumps and bare "No rows returned." to reach the user directly.
        # Force exactly one report_agent pass first, using whatever context
        # exists so far, then truly finish next time around.
        if state.get("next") != "report_agent":
            print(f"Hit max iterations ({MAX_ITERATIONS}) — forcing one report_agent pass instead of a raw dump.")
            return {"next": "report_agent", "iterations": iterations}
        else:
            print("Already forced a report_agent pass after max iterations — finishing now.")
            return {"next": "FINISH", "iterations": iterations}

    clean_context = build_sanitized_context(state["messages"])
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
# SQL Agent — now also uses sanitized context. It never needed to see
# raw tool_calls from EARLIER questions' SQL steps (each supervisor
# cycle only calls it once), so this is a pure safety improvement with
# no loss of needed reasoning.
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
    "'invoice_items'. For sales by genre, join Track -> InvoiceLine -> Genre.\n"
    "Always write a query that answers the user's MOST RECENT question in the "
    "conversation, not an earlier one.\n"
    "IMPORTANT: Country/location only exists on the Customer table (a customer's "
    "billing country) — there is NO country-of-origin field for tracks, artists, "
    "albums, or genres. If a question asks for something this schema has no column "
    "for, do NOT guess, search unrelated tables, or dump an entire table hoping for "
    "a match. Instead, respond in plain text (no tool call) explaining exactly what "
    "data is missing and why the question can't be answered from this schema."
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
    clean_context = build_sanitized_context(state["messages"])
    ai_msg = invoke_with_retry(
        sql_llm, [SQL_AGENT_SYSTEM_PROMPT] + clean_context, node_name="sql_agent"
    )
    new_messages = [ai_msg]
    for call in ai_msg.tool_calls:
        result = run_sql_query.invoke(call["args"])
        new_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": new_messages}


# ─────────────────────────────────────────────────────────────────
# Analysis + Report agents — now sanitized too. This directly fixes the
# raw "<|DSML|tool_calls>..." leak: with no raw tool_calls/ToolMessage
# objects in their context anymore, there's nothing left to imitate.
# ─────────────────────────────────────────────────────────────────
def analysis_agent_node(state: CrewState) -> dict:
    clean_context = build_sanitized_context(state["messages"])
    latest_question = get_latest_user_question(state["messages"])
    analysis_prompt = (
        f"The user's CURRENT question is: \"{latest_question}\"\n"
        "Review the SQL results in the conversation above and identify anything "
        "notable relevant to THIS question specifically. Be brief."
    )
    response = invoke_with_retry(
        llm, clean_context + [HumanMessage(analysis_prompt)], node_name="analysis_agent"
    )
    return {"messages": [AIMessage(content=response.content, name="analysis_agent")]}


def report_agent_node(state: CrewState) -> dict:
    clean_context = build_sanitized_context(state["messages"])
    latest_question = get_latest_user_question(state["messages"])
    report_prompt = (
        f"The user's CURRENT question is: \"{latest_question}\"\n"
        "Write a clear, final answer to THIS question specifically — do not "
        "re-answer an earlier question in the conversation. You may use context "
        "from earlier turns if relevant, but the current question is what you "
        "must directly answer. Be concise and cite the actual numbers."
    )
    response = invoke_with_retry(
        llm, clean_context + [HumanMessage(report_prompt)], node_name="report_agent"
    )
    return {"messages": [AIMessage(content=response.content, name="report_agent")]}


# ─────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────
def build_graph():
    """Builds and compiles the crew graph. Called once by the Streamlit app."""
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
    return builder.compile(checkpointer=checkpointer)