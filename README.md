# StatefulCrew

A multi-agent data analytics assistant built to learn LangChain and LangGraph end-to-end — from a single LLM call to a supervisor-coordinated crew of specialist agents with persistent state.

## Why this project exists

I'm a Data Engineer (AWS, Spark, SQL, dbt-style pipelines) learning how to build agentic AI systems on top of the data platforms I already work with. Instead of following a single tutorial, I built this project in phases — each one a self-contained step, each one committed separately — so the repo itself shows the progression from "LLM call" to "working multi-agent system."

The name **StatefulCrew** is literal: it's the one thing that separates LangGraph from a plain LangChain pipeline — a shared, persistent **state** object that a **crew** of agents reads from and writes to as they hand work back and forth.

## What it does

Ask a natural-language data question — e.g. *"What were the top 5 genres by total sales, and is there anything unusual about the numbers?"* — and a graph of agents handles it:

- **Supervisor** — decides which specialist acts next, or whether the task is done
- **SQL Agent** — writes and runs the actual database query, grounded in the real schema
- **Analysis Agent** — checks the results for patterns and anomalies
- **Report Agent** — writes the final answer, citing the actual numbers

All coordinated through a LangGraph `StateGraph`, with full execution traces in LangSmith.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| LLM | DeepSeek (`deepseek-v4-flash`) | Cheap enough to iterate freely while learning |
| Orchestration | LangChain + LangGraph | Industry-standard agent framework |
| Observability | LangSmith | See every agent step, not just the final answer |
| Database | SQLite (Chinook sample DB) | Zero-infra, focuses learning on the agents, not the infra |
| Interface | Streamlit | Simple demoable chat UI |
| Environment | Conda | Isolated, reproducible Python env |

## Project structure — one phase per folder

| Phase | Folder | What it covers |
|---|---|---|
| 1 | [`Phase1_Basics/`](./Phase1_Basics) | Plain LLM calls, prompt templates, structured output |
| 2 | [`Phase2_Tools/`](./Phase2_Tools) | Custom tools, SQL tool-calling, the ReAct loop |
| 3 | [`Phase3_LangGraphBasics/`](./Phase3_LangGraphBasics) | `StateGraph`, conditional edges, checkpointed memory |
| 4 | [`Phase4_MultiAgent/`](./Phase4_MultiAgent) | Supervisor pattern, the full agent crew |
| 5 | [`Phase5_Observability/`](./Phase5_Observability) | LangSmith tracing, guardrails, iteration limits |
| 6 | [`Phase6_Interface/`](./Phase6_Interface) | Streamlit front end |
| 7 | `Phase7_Deployment/` | Containerized, deployed on AWS |

Each phase folder has its own short README covering what was built and what broke along the way — debugging real provider quirks turned out to be most of the actual learning.

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/Suryaveer09/StatefulCrew.git
cd StatefulCrew

# Create and activate the conda environment
conda create -n data-insights-crew python=3.11 -y
conda activate data-insights-crew

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# then edit .env with your DEEPSEEK_API_KEY and LANGSMITH_API_KEY
```

## Environment variables

```
DEEPSEEK_API_KEY=your-deepseek-key
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=statefulcrew
```

## What I learned

- **DeepSeek's V4 models default to "thinking mode,"** which breaks LangChain's standard structured-output method (`with_structured_output`, function-calling variant). Worked around it in Phase 1 using `method="json_mode"` with an explicit schema description in the prompt.
- **`json_mode` is not reliable enough for every use case, even after tuning.** In Phase 4, the Supervisor's routing decision failed under `json_mode` in two different ways (empty content, then whitespace-only content) even after disabling thinking mode and setting `max_tokens`. The real fix was switching the routing mechanism to a tool call — the same mechanism that had been 100% reliable since Phase 2 — rather than continuing to tune a fragile approach. Lesson: when a fix changes a bug's *shape* without removing it, that's a signal to change strategy, not keep adjusting parameters.
- **Shared state in a multi-agent graph cuts both ways.** It's what makes agent handoff simple — every node sees the same state, no manual message-passing needed. But it also means a node can see things it was never meant to. In Phase 4, the Supervisor's own LLM call started imitating another agent's real tool-call JSON (`run_sql_query`) simply because that literal tool-call structure was sitting in the shared message history, even though `run_sql_query` was never bound to the Supervisor's own model. The fix was sanitizing what each node actually receives — stripping raw tool-call structures into plain-text summaries before they reach a node that shouldn't be acting on them. This was the single most non-obvious bug in the project so far: correct-looking code, wrong result, and the root cause was an architectural assumption (shared state = safe to share raw) rather than a syntax or config error.
- **A node's Python-level state and the actual API call are two different things to inspect.** Phase 5's LangSmith trace work confirmed the Phase 4 sanitization fix, but only after realizing that a node's own Input/Output tab always shows raw, unsanitized state — the real evidence is one level deeper, in the nested model-call span. Verifying a fix by reading a trace at the wrong level can *look* like the fix failed when it didn't.
- **A CLI test suite and a real UI test different things.** Building the Phase 6 Streamlit interface surfaced two bugs — dollar signs rendering as broken LaTeX, and every answer re-anchoring on the *first* question in a growing session instead of the current one — that six clean single-question CLI runs across Phases 4-5 never revealed. Both bugs only exist *because* of sustained multi-turn state and markdown rendering, neither of which the CLI script ever exercised.

Full writeups with the debugging process in each phase's own README — [`Phase1_Basics/README.md`](./Phase1_Basics/README.md), [`Phase2_Tools/README.md`](./Phase2_Tools/README.md), [`Phase3_LangGraphBasics/README.md`](./Phase3_LangGraphBasics/README.md), [`Phase4_MultiAgent/README.md`](./Phase4_MultiAgent/README.md), [`Phase5_Observability/README.md`](./Phase5_Observability/README.md), [`Phase6_Interface/README.md`](./Phase6_Interface/README.md).

## Status

🚧 In progress — currently on Phase 7 (Deployment).