# Phase 3 — LangGraph Basics

Goal: understand LangGraph's core primitives — State, nodes, edges, conditional routing, and checkpointing — by building up from the smallest possible graph to a full rebuild of the Phase 2 SQL agent, this time as an actual graph instead of a hand-written loop.

## What's in this folder

| File | What it demonstrates |
|---|---|
| `01_minimal_graph.py` | The four required pieces of every graph: State, a node, edges, `compile()` |
| `02_conditional_edge.py` | Branching — routing to a different node based on state |
| `03_sql_agent_graph.py` | The Phase 2 SQL agent rebuilt as a graph, with tool calling and persistent memory via a checkpointer |
| `chinook.db` *(not committed)* | Reused from Phase 2 — reads from `../Phase2_Tools/chinook.db` |

## Key concepts

**A graph has four required pieces:**
```python
class State(TypedDict):
    input: str
    output: str

def my_node(state: State) -> dict:
    return {"output": state["input"].upper()}  # only return what changed

builder = StateGraph(State)
builder.add_node("shout", my_node)
builder.add_edge(START, "shout")
builder.add_edge("shout", END)
graph = builder.compile()
```
Nodes are plain functions. They receive the full state but only need to return the fields they're updating — LangGraph merges the rest in automatically.

**Conditional edges route based on state, decided at runtime:**
```python
def route(state: State) -> Literal["even_node", "odd_node"]:
    return "even_node" if state["number"] % 2 == 0 else "odd_node"

builder.add_conditional_edges("check_number", route)
```
This is the exact mechanism, at its simplest, that later decides "call another tool" vs. "give a final answer" in a real agent.

**Reducers control how updates merge into state.** By default a node's return value *replaces* a state field. For a conversation, that's wrong — you want new messages appended, not overwriting history:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
`add_messages` is a reducer that appends instead of replaces. Every multi-turn agent graph needs this.

**`tools_condition` + a `tools → agent` edge replaces a manual while-loop:**
```python
builder.add_conditional_edges("agent", tools_condition)  # tool calls pending? → "tools" : → END
builder.add_edge("tools", "agent")                       # after tools run, let the agent decide what's next
```
This single pair of lines does exactly what the Phase 2 `while True` / `if not ai_msg.tool_calls: break` loop did by hand.

**Checkpointing gives a graph memory across calls, keyed by `thread_id`:**
```python
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "session-1"}}

graph.invoke({"messages": [...]}, config=config)   # turn 1
graph.invoke({"messages": [...]}, config=config)   # turn 2 — remembers turn 1 automatically
```

## What I verified — the graph vs. the manual loop

The main test for this phase was whether `03_sql_agent_graph.py` behaves identically to the Phase 2 hand-written loop, just restructured. Ran the same question through both:

> *"Which customer has spent the most money in total, and how much did they spend?"*

**Phase 2 (manual `while` loop):** `Helena Holý, $49.62`
**Phase 3 (LangGraph):** `Helena Holý, $49.62`

Same answer, same multi-round tool-calling behavior — confirms the graph version is a faithful structural replacement for the manual loop, not a behavior change.

**Then tested the checkpointer specifically** with a follow-up in the same thread:
```
Turn 1: "Which customer has spent the most money in total, and how much did they spend?"
→ Helena Holý, $49.62

Turn 2: "What city are they from?"
→ Helena Holý is from Prague, Czech Republic.
```
"they" resolved correctly with zero extra context sent — the checkpointer fed the prior turn back into state automatically. Using a new `thread_id` for the same follow-up question fails to resolve "they" at all, which confirms memory is properly scoped per-thread, not global.

## How to run

```bash
python 01_minimal_graph.py
python 02_conditional_edge.py
python 03_sql_agent_graph.py
```
(`03` requires `chinook.db` from Phase 2 to exist at `../Phase2_Tools/chinook.db`.)

## Why this matters for Phase 4

Phase 4's multi-agent crew is really just this same pattern — nodes, conditional edges, shared state — scaled up to more nodes: a Planner, a SQL Agent, an Analysis Agent, and a Report Agent, all reading and writing the same state object, with a supervisor node deciding who goes next instead of a simple tools_condition check.