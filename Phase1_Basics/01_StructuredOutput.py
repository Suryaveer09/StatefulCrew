# ─────────────────────────────────────────────────────────────────
# Phase 1: Structured Output — getting a typed object back instead
# of free text, using a prompt template to control exactly what
# gets classified.
# ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv

load_dotenv()  # pulls DEEPSEEK_API_KEY and LANGSMITH_* vars from .env into the environment

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)


# Pydantic defines the exact SHAPE we want back from the model — field
# names, types, and descriptions the model uses to understand what each
# field means. This becomes the schema the model is asked to fill in.
class DataQuestionAnalysis(BaseModel):
    """Structured breakdown of a user's data question."""

    requires_database: bool = Field(description="Does this question need a SQL query?")
    requires_analysis: bool = Field(
        description="Does this question need statistical analysis?"
    )
    complexity: str = Field(description="One of: simple, moderate, complex")
    reasoning: str = Field(description="Brief explanation of the classification")


# with_structured_output binds the schema to the model. method="json_mode"
# is used here instead of the default "function_calling" — DeepSeek's V4
# models run in "thinking mode" by default, which rejects the forced
# tool_choice that the default method relies on. json_mode sidesteps that,
# but in exchange it does NOT auto-inject the schema into the model's
# awareness the way function_calling normally would — the prompt below
# has to describe the required fields explicitly instead.
structured_llm = llm.with_structured_output(DataQuestionAnalysis, method="json_mode")

# The system message does two jobs at once:
# 1. Tells the model to classify, not answer — without this, the model
#    will try to actually solve the user's question instead of describing it
# 2. Spells out the exact field names/types expected, compensating for
#    json_mode not seeing the Pydantic schema directly
# The word "json" appearing in the prompt is also a DeepSeek API
# requirement for JSON mode specifically, not just a LangChain convention.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing classifier. Do NOT answer the user's question or invent any data. "
            "Your only job is to classify the question and return json with exactly these fields: "
            "requires_database (bool), requires_analysis (bool), "
            "complexity (one of: simple, moderate, complex), reasoning (short string).",
        ),
        ("human", "{question}"),
    ]
)

# LCEL chain: prompt output feeds directly into the structured model call
chain = prompt | structured_llm

# result is a real DataQuestionAnalysis object — not a string to parse,
# not a dict to index into. Pydantic has already validated the types.
result = chain.invoke(
    {
        "question": "What were the top 5 products by revenue last quarter, and are there any anomalies?"
    }
)

print(result.requires_database)
print(result.requires_analysis)
print(result.complexity)
print(result.reasoning)