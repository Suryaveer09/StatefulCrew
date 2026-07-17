# StatefulCrew

A multi-agent data analytics assistant built to learn LangChain and LangGraph end-to-end — from a single LLM call to a supervisor-coordinated crew of specialist agents, deployed live on Azure.

**Live demo:** `https://statefulcrew-app.azurewebsites.net` *(stopped by default to avoid ongoing cost — see [Phase7_Deployment/README.md](./Phase7_Deployment/README.md) for how to start it)*

## Why this project exists

I'm a Data Engineer (AWS, Spark, SQL, dbt-style pipelines) learning how to build agentic AI systems on top of the data platforms I already work with. Instead of following a single tutorial, I built this project in phases — each one a self-contained step, each one committed separately — so the repo itself shows the progression from "LLM call" to a working, deployed multi-agent system.

The name **StatefulCrew** is literal: it's the one thing that separates LangGraph from a plain LangChain pipeline — a shared, persistent **state** object that a **crew** of agents reads from and writes to as they hand work back and forth.

## What it does

Ask a natural-language data question — e.g. *"What were the top 5 genres by total sales, and is there anything unusual about the numbers?"* — and a graph of agents handles it:

- **Supervisor** — decides which specialist acts next, or whether the task is done
- **SQL Agent** — writes and runs the actual database query, grounded in the real schema, honest when the data can't answer the question
- **Analysis Agent** — checks the results for patterns and anomalies
- **Report Agent** — writes the final answer, citing the actual numbers

All coordinated through a LangGraph `StateGraph`, with full execution traces in LangSmith, wrapped in a Streamlit chat UI, containerized with Docker, and deployed on Azure.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| LLM | DeepSeek (`deepseek-v4-flash`) | Cheap enough to iterate freely while learning |
| Orchestration | LangChain + LangGraph | Industry-standard agent framework |
| Observability | LangSmith | See every agent step, not just the final answer |
| Database | SQLite (Chinook sample DB) | Zero-infra, focuses learning on the agents, not the infra |
| Interface | Streamlit | Chat UI, containerized with Docker |
| Cloud | Azure (App Service + Container Registry) | Targeted deliberately — matches the Azure requirement in the roles I'm applying to, distinct from my existing AWS background |
| Environment | Conda | Isolated, reproducible Python env |

## Project structure — one phase per folder

| Phase | Folder | What it covers |
|---|---|---|
| 1 | [`Phase1_Basics/`](./Phase1_Basics) | Plain LLM calls, prompt templates, structured output |
| 2 | [`Phase2_Tools/`](./Phase2_Tools) | Custom tools, SQL tool-calling, the ReAct loop |
| 3 | [`Phase3_LangGraphBasics/`](./Phase3_LangGraphBasics) | `StateGraph`, conditional edges, checkpointed memory |
| 4 | [`Phase4_MultiAgent/`](./Phase4_MultiAgent) | Supervisor pattern, the full agent crew |
| 5 | [`Phase5_Observability/`](./Phase5_Observability) | LangSmith tracing, guardrails, iteration limits |
| 6 | [`Phase6_Interface/`](./Phase6_Interface) | Streamlit front end, multi-turn conversation bugs |
| 7 | [`Phase7_Deployment/`](./Phase7_Deployment) | Docker containerization, Azure deployment |

Each phase folder has its own README covering what was built and what broke along the way — debugging real provider quirks and cloud deployment issues turned out to be most of the actual learning.

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

To run the deployed version locally instead: see [`Phase7_Deployment/README.md`](./Phase7_Deployment/README.md) for Docker build/run instructions.

## Environment variables

```
DEEPSEEK_API_KEY=your-deepseek-key
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=statefulcrew
```

## What I learned

- **DeepSeek's V4 models default to "thinking mode,"** which breaks LangChain's standard structured-output method. Worked around it in Phase 1 using `method="json_mode"` with an explicit schema description in the prompt.
- **`json_mode` is not reliable enough for every use case, even after tuning.** In Phase 4, the Supervisor's routing decision failed under `json_mode` in two different ways even after disabling thinking mode and setting `max_tokens`. The real fix was switching to a tool call — the same mechanism that had been 100% reliable since Phase 2. When a fix changes a bug's *shape* without removing it, that's a signal to change strategy, not keep adjusting parameters.
- **Shared state in a multi-agent graph cuts both ways.** It's what makes agent handoff simple, but it also means a node can see things it was never meant to. The Supervisor's own LLM call started imitating another agent's real tool-call JSON simply because that structure was sitting in shared history. The fix — sanitizing what each node actually receives — was the single most non-obvious bug in the project: correct-looking code, wrong result, and the root cause was an architectural assumption, not a syntax error.
- **A node's Python-level state and the actual API call are two different things to inspect.** Verifying the sanitization fix via LangSmith required going one level deeper than a node's own Input/Output tab, into the nested model-call span — the node's own state view always looks unsanitized by design.
- **A CLI test suite and a real UI test different things.** Building the Streamlit interface surfaced bugs — dollar signs rendering as broken LaTeX, answers re-anchoring on the first question in a growing session — that six clean single-question CLI runs never revealed, because neither bug could exist without sustained multi-turn state.
- **Fixing one bug can unmask the next layer down.** Sanitizing context (fixing a leak) exposed a missing graceful-failure path, and fixing *that* broke a completely separate, un-synced guardrail (LangGraph's `recursion_limit` vs. the crew's own `MAX_ITERATIONS`). None of this meant the approach was wrong — it's the normal shape of debugging a system with several interacting safety mechanisms. Deriving one guardrail from the other, instead of hardcoding both separately, closed that gap for good.
- **A working local Docker container doesn't guarantee a working cloud deployment.** Every actual Azure deployment bug — unregistered resource providers, a transient resource lock, a placeholder API key that slipped through — was specific to the cloud environment and invisible to local testing, even after thorough local validation.
- **Testing for a limitation and documenting it honestly beats hiding it.** A deliberate prompt-injection test on the live Azure deployment found a real gap — the crew complied with "ignore previous instructions" instead of staying on-task. Left undocumented, that's a landmine for later. Written up openly, it's evidence of genuine adversarial testing rather than a suspiciously perfect record.

Full writeups with the debugging process in each phase's own README — [`Phase1_Basics/README.md`](./Phase1_Basics/README.md), [`Phase2_Tools/README.md`](./Phase2_Tools/README.md), [`Phase3_LangGraphBasics/README.md`](./Phase3_LangGraphBasics/README.md), [`Phase4_MultiAgent/README.md`](./Phase4_MultiAgent/README.md), [`Phase5_Observability/README.md`](./Phase5_Observability/README.md), [`Phase6_Interface/README.md`](./Phase6_Interface/README.md), [`Phase7_Deployment/README.md`](./Phase7_Deployment/README.md).

## Known limitations

- **No defense against prompt injection or off-topic instructions.** Found via deliberate adversarial testing in Phase 7 — no system prompt currently tells any agent how to handle a request that tries to override its instructions. Not fixed, by choice, to keep this phase focused on deployment rather than opening a new prompt-hardening phase — but documented rather than hidden.
- **SQLite, not a production database.** Chosen deliberately to keep the learning focus on the agent architecture rather than database infrastructure. A "real" production version would use Azure SQL or PostgreSQL.
- **Single-model setup.** Every agent uses the same `deepseek-v4-flash` model. A more cost/quality-optimized version might use a smaller model for routing and a stronger one for analysis/report writing.

## Status

✅ Core project complete — Phases 1 through 7 built, debugged, and deployed. The live app is stopped by default (see Phase 7's README for cost details and how to start it for a demo).