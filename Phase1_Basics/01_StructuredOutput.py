from dotenv import load_dotenv

load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)


class DataQuestionAnalysis(BaseModel):
    """Structured breakdown of a user's data question."""

    requires_database: bool = Field(description="Does this question need a SQL query?")
    requires_analysis: bool = Field(
        description="Does this question need statistical analysis?"
    )
    complexity: str = Field(description="One of: simple, moderate, complex")
    reasoning: str = Field(description="Brief explanation of the classification")


structured_llm = llm.with_structured_output(DataQuestionAnalysis, method="json_mode")

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

chain = prompt | structured_llm

result = chain.invoke(
    {
        "question": "What were the top 5 products by revenue last quarter, and are there any anomalies?"
    }
)

print(result.requires_database)
print(result.requires_analysis)
print(result.complexity)
print(result.reasoning)
