"""Basic Prompt Template - Reusable, Structured Prompts
Instead of hardcoding strings, we separate the template from the variables we fill in:
"""
from dotenv import load_dotenv

load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)

# A template with a system role (sets behavior) and a human role (the actual ask)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a senior data analyst. Be concise and precise."),
        ("human", "Explain what {concept} means in the context of {domain}."),
    ]
)

# Chains combine steps with the | (pipe) operator — this is LangChain's core pattern
chain = prompt | llm

response = chain.invoke(
    {"concept": "partitioning", "domain": "distributed data processing"}
)
print(response.content)
