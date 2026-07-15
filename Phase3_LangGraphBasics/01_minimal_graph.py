# ─────────────────────────────────────────────────────────────────
# Phase 3, Step 1: The smallest possible LangGraph
#
# Goal: see the four required pieces of every LangGraph — State,
# nodes, edges, and compile() — before adding any real complexity
# like branching or tool calls.
# ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# State is a shared data structure that flows through every node in
# the graph. Every node receives the current state and returns updates
# to it. Here it's just two plain strings — no reducers or message
# lists needed yet, since there's nothing to accumulate.
class State(TypedDict):
    input: str
    output: str


def uppercase_node(state: State) -> dict:
    """A node is just a Python function: it takes the current state and
    returns a dict of the fields it wants to update. It does NOT need
    to return the whole state — only the parts that changed.
    """
    return {"output": state["input"].upper()}


# ─────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────
builder = StateGraph(State)

# Register the node under the name "shout" — this name is how edges
# will refer to it below.
builder.add_node("shout", uppercase_node)

# START and END are special built-in markers, not real nodes. Every
# graph needs at least one edge from START (where execution begins)
# and one edge into END (where execution stops).
builder.add_edge(START, "shout")
builder.add_edge("shout", END)

# The builder can't run anything on its own — .compile() validates the
# graph structure and turns it into an executable object.
graph = builder.compile()


# ─────────────────────────────────────────────────────────────────
# Run it
# ─────────────────────────────────────────────────────────────────
# .invoke() takes an initial state (just needs to satisfy the fields
# the graph will read — "output" doesn't need to be provided since no
# node reads it before writing it) and runs the graph start to finish.
result = graph.invoke({"input": "hello statefulcrew"})
print(result)
# Expected: {'input': 'hello statefulcrew', 'output': 'HELLO STATEFULCREW'}
