# ─────────────────────────────────────────────────────────────────
# Phase 6: Streamlit Interface — wraps the crew (imported from
# crew.py, not run standalone) in a real chat UI with per-session
# memory isolation and safe markdown rendering.
# ─────────────────────────────────────────────────────────────────

import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

# Importing build_graph() rather than a script that runs itself — crew.py
# only builds and returns the compiled graph, it never invokes it directly
from crew import build_graph, MAX_ITERATIONS

# LangGraph's recursion_limit counts EVERY node execution (supervisor AND
# specialists), not just supervisor iterations like our own MAX_ITERATIONS
# does. Each supervisor->specialist round is 2 graph steps, and the forced
# report_agent pass after hitting MAX_ITERATIONS adds a few more on top.
# Deriving this from MAX_ITERATIONS (instead of a separate hardcoded
# number) means the two guardrails can never drift out of sync again —
# that mismatch is exactly what caused a GraphRecursionError here.
RECURSION_LIMIT = (MAX_ITERATIONS * 2) + 5

st.set_page_config(page_title="StatefulCrew", page_icon="🎵")
st.title("🎵 StatefulCrew — Data Insights Assistant")
st.caption(
    "Ask a question about the Chinook music store database. A multi-agent crew "
    "(Supervisor → SQL Agent → Analysis Agent → Report Agent) will work it out."
)


def safe_markdown(text: str):
    """Streamlit renders a single $ as the start of LaTeX math by default,
    which mangles dollar amounts in the crew's answers (e.g. "$93.53" next
    to "**bold**" gets parsed as math instead of displayed as text).
    Escaping every $ before rendering fixes this. Found via Phase 6 UI testing.
    """
    st.markdown(text.replace("$", "\\$"))


# The graph is expensive to rebuild on every interaction — cache it so
# Streamlit only builds it once per session, not on every widget rerun.
# Without this, every button click / chat submission would silently
# rebuild the whole graph (and its checkpointer) from scratch.
@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

# Each browser session gets its own thread_id, so conversations don't
# bleed into each other across different users/tabs. Generated once per
# session and stored in st.session_state, which persists across Streamlit's
# repeated script reruns (Streamlit re-executes this whole file on every
# interaction — session_state is what survives that).
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# display_history is separate from the graph's own checkpointed state —
# this is purely what gets RENDERED in the chat UI, a simple list of
# (role, text) tuples.
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# Render every prior turn on each rerun, since Streamlit doesn't
# persist rendered UI across reruns — it has to be redrawn every time
for role, content in st.session_state.display_history:
    with st.chat_message(role):
        safe_markdown(content)

# The chat input widget — returns the typed message once submitted,
# otherwise returns None (so the block below only runs on real input)
question = st.chat_input("Ask about the music store data...")

if question:
    # Add and immediately render the user's own message
    st.session_state.display_history.append(("user", question))
    with st.chat_message("user"):
        safe_markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Crew is working..."):
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }
            # Only the NEW human message is sent — the checkpointer
            # (keyed by thread_id) supplies all prior conversation
            # history automatically, so we never resend the full history
            result = graph.invoke(
                {"messages": [HumanMessage(question)], "iterations": 0},
                config=config,
            )
            answer = result["messages"][-1].content
        safe_markdown(answer)

    # Record the assistant's answer so it's still shown after the next rerun
    st.session_state.display_history.append(("assistant", answer))