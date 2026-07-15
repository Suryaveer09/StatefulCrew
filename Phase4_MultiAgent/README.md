# Phase 4 — Multi-Agent Crew (Supervisor Pattern)

Goal: turn the single SQL agent from Phase 3 into a coordinated crew — a Supervisor that routes work to specialist agents (SQL Agent, Analysis Agent, Report Agent), each handing control back to the Supervisor until the task is done.

This phase produced the hardest debugging of the whole project — five distinct bugs across two categories: getting the graph to *run* at all, and then getting it to run *correctly*. Both are documented in full below, since the debugging process is honestly the most valuable part of this phase.

## What's in this folder

| File | What it demonstrates |
|---|---|
| `multi_agent_crew.py` | The full supervisor + 3-specialist crew: routing via tool calls, schema-grounded SQL agent, sanitized inter-agent context |

## Architecture

```
        ┌─────────────┐
        │  Supervisor │◄───────────────┐
        └──────┬──────┘                │
   ┌────────────┼────────────┐         │
   ▼            ▼             ▼        │
┌────────┐ ┌──────────┐ ┌──────────┐   │
│  SQL   │ │ Analysis │ │  Report  │   │
│ Agent  │ │  Agent   │ │  Agent   │   │
└───┬────┘ └────┬─────┘ └────┬─────┘   │
    └────────────┴────────────┴────────┘
```

The Supervisor is an LLM that looks at shared state and decides which specialist runs next, or whether the task is done. Every specialist hands control back to the Supervisor rather than answering the user directly — this is the standard "supervisor pattern," chosen over more complex alternatives (swarm/peer-to-peer handoffs) because it's easier to reason about and debug: one routing decision-maker, one place to look when something goes wrong.

## Key concepts

**Shared state, not shared memory.** Every node — supervisor and all three specialists — reads from and writes to the same `CrewState`. That's what makes "handing off" possible: the SQL Agent's tool results are automatically visible to the Analysis Agent without any agent-to-agent messaging code.

**Routing as a tool call, not structured output.** The Supervisor decides "who's next" by calling a `route(next_agent: Literal[...])` tool rather than using `with_structured_output()`. This was a deliberate pivot after `json_mode` proved unreliable for this specific node — full story below.

## What broke — five bugs, two categories

### Category 1: Getting the graph to run without crashing

**Bug 1 — Definition-before-use ordering.** `builder = StateGraph(CrewState)` got placed before `CrewState` was defined, and before any node functions existed. Python executes top-to-bottom; `StateGraph(...)` and every `add_node()`/`add_edge()` call must come *after* everything they reference.
> Fix: moved graph construction to the bottom of the file, after every node function.

**Bug 2 — DeepSeek's `json_mode` failing in two different ways.** The Supervisor used `with_structured_output(RouteDecision, method="json_mode")` to pick the next agent. First failure: consistent empty content (`Expecting value: line 1 column 1 (char 0)`) — not intermittent, 3/3 retries failed identically. Hypothesized thinking-mode token exhaustion (DeepSeek's chain-of-thought competing with the actual answer for the same token budget) and disabled thinking mode + set `max_tokens=512`. That changed the failure, but didn't fix it: second failure was **135 tokens of pure whitespace**, `finish_reason: 'stop'` (not a token-limit truncation — the model was actively choosing to emit blank content).
> Fix: stopped trying to tune `json_mode` configuration and switched the Supervisor's routing mechanism to a **tool call** instead — the same mechanism that had been 100% reliable since Phase 2's SQL agent. Lesson: when a "fix" changes a bug's *shape* without eliminating it, that's a signal to change approach, not keep tuning parameters.

**Bug 3 — Tool argument key mismatch.** After switching to tool-calling, `ai_msg.tool_calls[0]["args"]["next"]` raised `KeyError: 'next'`. Root cause never fully confirmed, but `next` shadowing Python's builtin `next()` was a reasonable suspect worth eliminating regardless.
> Fix: renamed the tool parameter to `next_agent`, and made `supervisor_node` defensive — print the actual args and fall back to `FINISH` instead of crashing if the expected key is ever missing again.

### Category 2: Getting the graph to run *correctly* (no crash, wrong answer)

This is the harder category — the graph completed successfully but produced a garbage final answer. Diagnosis required reading the full trace carefully rather than a stack trace pointing at a line number.

**Bug 4 — The Supervisor imitated another agent's tool call.** With crashing fixed, a defensive print revealed the Supervisor's own LLM call had requested `run_sql_query` — a tool never bound to it (`router_llm` only knows about `route`). Root cause: `state["messages"]` is shared across the whole crew, so by the Supervisor's second turn, its context already contained the SQL Agent's real `run_sql_query` tool-call JSON from earlier in the loop. DeepSeek appears to have pattern-matched and imitated that literal tool-call structure instead of respecting its own actual tool schema.
> Fix: built `build_supervisor_context()` — strips raw `tool_calls` and `ToolMessage` objects out of what the Supervisor sees, replacing them with plain-text summaries (`"[sql_agent used a tool: run_sql_query]"`) before they ever reach its prompt. No literal tool-call JSON from other agents = nothing to imitate.

**Bug 5 — The SQL Agent hallucinated a table name.** `SQL Error: no such table: invoice_items`. The real Chinook table is `InvoiceLine`, not `invoice_items` — a plausible-sounding but incorrect guess, since the SQL Agent had zero schema information and was working entirely from training-data pattern matching about "typical" schemas. (The `report_agent`'s final answer had simply echoed this raw SQL error back as its "report" — not a crash, a complete but useless run.)
> Fix: added an explicit `SQL_AGENT_SYSTEM_PROMPT` listing the real table and column names, prepended to every SQL Agent call. Standard real-world SQL agent practice: ground the model in the actual schema rather than letting it guess.

## Result after all five fixes

Question: *"What were the top 5 genres by total sales, and is there anything unusual about the numbers?"*

```
1. Rock – $826.65
2. Latin – $382.14
3. Metal – $261.36
4. Alternative & Punk – $241.56
5. TV Shows – $93.53

What's unusual?
- TV Shows isn't a music genre, yet it ranks in the top 5
- Latin outsells Metal by more than 2x — unexpectedly strong demand
- Steep drop-off after #4, revealing a long tail of low-selling genres
```

Correct numbers, no misrouting, no crashes — and the Analysis Agent's "TV Shows isn't a music genre" observation is a genuinely sharp catch, not something explicitly prompted for.

## How to run

```bash
python multi_agent_crew.py
```
(Requires `chinook.db` from Phase 2 at `../Phase2_Tools/chinook.db`.)

## Takeaways for Phase 5+

- **Two categories of bug matter differently.** Crash bugs (1-3) are loud and easy to notice. Correctness bugs (4-5) are quiet — the system "works" and still gives a wrong answer. Phase 5's observability work (LangSmith trace inspection, guardrails) exists specifically to catch category 2 faster than "read the final answer and notice it's wrong."
- **Shared state cuts both ways.** It's what makes multi-agent handoff simple, but it also means every node sees everything — including things that can confuse a model that wasn't meant to see them. Sanitizing what each node actually receives is a real design decision, not an edge case.
- **When a fix changes a bug's shape instead of removing it, that's a signal.** Two different `json_mode` configurations produced two different failures from the same underlying fragility — the right move was switching mechanisms entirely, not further parameter tuning.