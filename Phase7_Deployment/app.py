import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

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
@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

# Each browser session gets its own thread_id, so conversations don't
# bleed into each other across different users/tabs.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_history" not in st.session_state:
    st.session_state.display_history = []  # what we SHOW the user

# Render prior turns
for role, content in st.session_state.display_history:
    with st.chat_message(role):
        safe_markdown(content)

# New user input
question = st.chat_input("Ask about the music store data...")

if question:
    st.session_state.display_history.append(("user", question))
    with st.chat_message("user"):
        safe_markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Crew is working..."):
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }
            result = graph.invoke(
                {"messages": [HumanMessage(question)], "iterations": 0},
                config=config,
            )
            answer = result["messages"][-1].content
        safe_markdown(answer)

    st.session_state.display_history.append(("assistant", answer))