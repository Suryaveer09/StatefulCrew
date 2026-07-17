# ─────────────────────────────────────────────────────────────────
# Phase 1: Basic Prompt Template — Reusable, Structured Prompts
#
# Instead of hardcoding strings, we separate the template from the
# variables we fill in, and combine steps into a chain using LCEL
# (LangChain Expression Language).
# ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv

load_dotenv()  # pulls DEEPSEEK_API_KEY and LANGSMITH_* vars from .env into the environment

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)

# A template with a system role (sets behavior) and a human role (the
# actual ask). The {concept} and {domain} placeholders get filled in at
# call time — the template itself is reusable across many different
# concept/domain pairs without rewriting the prompt each time.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a senior data analyst. Be concise and precise."),
        ("human", "Explain what {concept} means in the context of {domain}."),
    ]
)

# Chains combine steps with the | (pipe) operator — this is LangChain's
# core pattern (LCEL). Here it means: take the filled-in prompt, feed it
# directly into the LLM. This same "pipe steps together" mental model
# carries over conceptually to how LangGraph nodes pass state later,
# even though the actual API is different.
chain = prompt | llm

# invoke() supplies the values for the template's placeholders. LangChain
# fills in {concept} and {domain}, builds the final messages, and sends
# them to the model in one call.
response = chain.invoke(
    {"concept": "partitioning", "domain": "distributed data processing"}
)

# response is an AIMessage object — .content pulls out just the text
print(response.content)