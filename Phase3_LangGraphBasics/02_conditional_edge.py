# ─────────────────────────────────────────────────────────────────
# Phase 3, Step 2: Conditional edges — routing based on state
#
# Goal: see how a graph can branch down different paths depending on
# the data in state. This is the mechanism that later lets an agent
# decide "call a tool" vs "give a final answer" — the exact same
# pattern, just with a trivial even/odd check instead of a real
# decision.
# ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    number: int
    result: str


def check_number(state: State) -> dict:
    """This node exists purely as a routing point in the graph — it
    doesn't need to change any state itself. The actual decision
    happens in the route() function below, which runs right after
    this node completes.
    """
    return {}


def even_node(state: State) -> dict:
    return {"result": f"{state['number']} is even"}


def odd_node(state: State) -> dict:
    return {"result": f"{state['number']} is odd"}


def route(state: State) -> Literal["even_node", "odd_node"]:
    """A conditional edge function: it reads the current state and
    returns the NAME of whichever node should run next. LangGraph
    calls this automatically after "check_number" finishes, and sends
    execution to whichever node name gets returned.
    """
    return "even_node" if state["number"] % 2 == 0 else "odd_node"


# ─────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("check_number", check_number)
builder.add_node("even_node", even_node)
builder.add_node("odd_node", odd_node)

builder.add_edge(START, "check_number")

# add_conditional_edges wires up branching: after "check_number" runs,
# call route(state) and send execution to whatever node name it
# returns. Unlike add_edge, this isn't a fixed path — the destination
# is decided at runtime based on state.
builder.add_conditional_edges("check_number", route)

# Both branches terminate the graph the same way
builder.add_edge("even_node", END)
builder.add_edge("odd_node", END)

graph = builder.compile()


# ─────────────────────────────────────────────────────────────────
# Run it with two different inputs to see both branches fire
# ─────────────────────────────────────────────────────────────────
print(graph.invoke({"number": 7}))   # {'number': 7, 'result': '7 is odd'}
print(graph.invoke({"number": 4}))   # {'number': 4, 'result': '4 is even'}
