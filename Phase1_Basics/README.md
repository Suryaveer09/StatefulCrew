# Phase 1 — LangChain Basics

Goal: get comfortable with the core building blocks before touching agents or LangGraph — a plain LLM call, prompt templates, and structured output.

## What's in this folder

| File | What it demonstrates |
|---|---|
| `BasicPromptTemplate.py` | `ChatPromptTemplate` + the `\|` (LCEL) chain pattern |
| `StructuredOutput.py` | Getting typed, validated Pydantic objects back instead of raw text |

## Key concepts

**Prompt templates** separate the fixed instruction (system role) from the variable input (human role), and get filled in at call time:
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior data analyst. Be concise and precise."),
    ("human", "Explain what {concept} means in the context of {domain}.")
])
chain = prompt | llm
```
The `|` operator is LangChain Expression Language (LCEL) — the same "pipe steps together" mental model carries over conceptually to how LangGraph nodes pass state, even though the API is different.

**Structured output** turns free text into a real, typed object:
```python
class DataQuestionAnalysis(BaseModel):
    requires_database: bool
    requires_analysis: bool
    complexity: str
    reasoning: str

structured_llm = llm.with_structured_output(DataQuestionAnalysis)
```
This is the exact pattern the Planner agent uses in Phase 4 to decide how to route a question to the right specialist.

## What broke (and why it mattered)

DeepSeek's V4 models run in "thinking mode" by default, which rejects the forced `tool_choice` call that LangChain's default `with_structured_output()` method relies on:

```
openai.BadRequestError: Error code: 400 - {'error': {'message': 'Thinking mode does not support this tool_choice', ...}}
```

**Fix:** use `method="json_mode"` instead of the default `"function_calling"`:
```python
structured_llm = llm.with_structured_output(DataQuestionAnalysis, method="json_mode")
```

That surfaced a second, more interesting problem: in `json_mode`, DeepSeek never sees your Pydantic schema — it only knows "return valid JSON." Without an explicit instruction, it answered the *business question* instead of classifying it, returning fabricated product revenue data instead of the four expected fields.

**Fix:** be explicit in the system prompt about the task *and* the exact field names/types expected:
```python
("system",
 "You are a routing classifier. Do NOT answer the user's question or invent any data. "
 "Your only job is to classify the question and return json with exactly these fields: "
 "requires_database (bool), requires_analysis (bool), "
 "complexity (one of: simple, moderate, complex), reasoning (short string).")
```

**Takeaway:** `json_mode` trades away the automatic schema-awareness that `function_calling` mode normally provides — you have to describe the shape yourself. Worth remembering for every structured-output call in the rest of this project, since the Planner, SQL Agent, and Report Agent will all lean on this same pattern.

## How to run

```bash
python BasicPromptTemplate.py
python StructuredOutput.py
```

## Result

```
requires_database: True
requires_analysis:  True
complexity:         moderate
reasoning:          Retrieves product revenue and identifies anomalies
```