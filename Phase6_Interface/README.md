# Phase 6 — Streamlit Interface

Goal: wrap the Phase 5 crew in a real, demoable chat UI instead of a one-shot CLI script — and along the way, stress-test it with genuine multi-turn conversation, which surfaced two bugs that never showed up in any single-question CLI run.

## What's in this folder

| File | What it demonstrates |
|---|---|
| `crew.py` | The crew as an importable module — `build_graph()` instead of a script that runs itself on import |
| `app.py` | The Streamlit chat UI: session-scoped conversations, cached graph construction, safe markdown rendering |
| `images/dollar_sign_rendering_bug.png` | Evidence of Bug A — see below |
| `images/stale_question_bug.png` | Evidence of Bug B — see below |
| `images/fixed_output.png` | Confirmation both bugs are fixed, with a genuinely new finding from the same question |

## Key concepts

**Making a script importable vs. runnable.** Every prior phase's script ran itself top-to-bottom via `graph.invoke(...)` at the bottom of the file. Streamlit needs to *import* the graph-building logic without that code executing on import — so the graph construction moved into a `build_graph()` function, called explicitly by `app.py` rather than running automatically.

**Session-scoped state via `thread_id`.** Each browser session gets its own `uuid4()` stored in `st.session_state`, used as the LangGraph checkpointer's `thread_id`. This is what keeps two browser tabs from bleeding into each other's conversation — same mechanism from Phase 3's memory demo, now driving real multi-user isolation.

**Caching the graph, not rebuilding it per interaction.** `@st.cache_resource` ensures `build_graph()` runs once per session, not on every widget rerun — Streamlit re-executes the whole script on every interaction, so without this the graph (and its checkpointer) would be rebuilt from scratch on every single message, silently discarding memory.

## What broke — two bugs only a real UI could surface

Six single-shot CLI questions in Phases 4-5 never revealed either of these. Both only showed up once real multi-turn conversation and markdown rendering were actually exercised — a good reminder that a UI isn't just packaging, it's a genuine new test surface.

**Bug A — dollar amounts rendering as garbled LaTeX.** Streamlit's `st.markdown()` treats a single `$` as the start of inline math by default. With dollar amounts and `**bold**` markdown both present in the same answer, two `$` symbols anywhere in a block got interpreted as opening/closing a math expression, mangling numbers and swallowing the bold markers:

![Dollar sign rendering bug](./images/dollar_sign_rendering_bug.png)

> Fix: added `safe_markdown()` — escapes every `$` to `\$` before rendering. Applied everywhere the app calls `st.markdown()` on model or user text.

**Bug B — every answer re-anchored on the first question in the session, not the current one.** By the third or fourth turn in a session, every answer led with a rehash of the very first question's genre table, with the actual new question's answer buried as a sub-bullet or ignored entirely:

![Stale question bug](./images/stale_question_bug.png)

Root cause: `report_agent_node`'s prompt said *"answer the user's **original** question"* — a phrase that reads fine for a single CLI call, but once `state["messages"]` accumulates a real multi-turn history (every prior Q&A stays in state under the same `thread_id`), "original question" naturally resolves to the literal first `HumanMessage` in the growing list, not the current one.

> Fix: added `get_latest_user_question()` — walks `state["messages"]` in reverse to find the most recent `HumanMessage`, and both `analysis_agent_node` and `report_agent_node` now explicitly quote that exact question in their prompts instead of relying on the ambiguous word "original."

## Result after both fixes

Re-ran the same 6-question sequence. Every answer now correctly addresses its own question, and `$` amounts render cleanly:

![Fixed output — new finding, not a rehash](./images/fixed_output.png)

The clearest evidence the fix actually changed model behavior (not just UI rendering): the open-ended "tell me something interesting" question produced a **different, better** answer on the fixed run — *"43% of the catalog (1,519 of 3,503 tracks) has never sold a single unit"* — instead of defaulting back to the genre table it kept re-deriving before. Fixing the stale-question bug didn't just patch a symptom; it let the Report Agent actually explore new territory instead of being anchored to the first question's context.

## How to run

```bash
pip install streamlit
streamlit run app.py
```
Requires `chinook.db` from Phase 2 at `../Phase2_Tools/chinook.db`. Opens automatically at `localhost:8501`.

## Takeaways for Phase 7

- **A CLI test suite and a real UI test different things.** Six clean single-question CLI runs across Phases 4-5 gave no signal that multi-turn context or markdown rendering were broken — both bugs only exist *because* of sustained, stateful conversation, which the CLI script never exercised (it called `graph.invoke()` exactly once per run).
- **Ambiguous prompt language is a real bug class.** "The user's original question" wasn't wrong when the state was short — it became wrong as state grew. Any prompt that references "the question," "the task," or "the goal" without pinning down *which* one is worth re-checking once multi-turn state is in play.
- **Session isolation (`thread_id` per tab) is cheap to build and easy to forget.** One `uuid4()` in `st.session_state` is the entire mechanism — worth verifying explicitly (two tabs, two separate conversations) rather than assuming it works because the code looks right.