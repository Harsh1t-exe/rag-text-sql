"""
Text-to-SQL generation via Groq's free API (https://console.groq.com).
Groq offers a free tier with no credit card required and serves fast,
open-source models on their LPU hardware - good accuracy on SQL generation
without running your own GPU server.

Get a free API key at https://console.groq.com/keys and set it as the
GROQ_API_KEY environment variable.

This module builds prompts from RETRIEVED context (relevant tables + similar
past examples) rather than the full schema, and implements a self-correction
loop: if a generated query fails to execute, the error is fed back to the
model for up to MAX_RETRIES attempts before giving up.
"""
import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq deprecated the llama-3.3-70b-versatile family. gpt-oss-120b is the
# current recommended general-purpose/reasoning model on Groq's free tier.
# If this model is ever deprecated too, check https://console.groq.com/docs/models
# or GET https://api.groq.com/openai/v1/models with your API key for the live list.
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are an expert SQL generator. Given a set of relevant
database tables and a natural language question, output ONLY a single valid,
read-only SQL SELECT query that answers the question.

Rules:
- Output ONLY the SQL query, no explanation, no markdown code fences, no comments.
- Only generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
- Use only the tables and columns shown below - do not invent tables or columns.
- Prefer explicit column names over SELECT *.
- Add a reasonable LIMIT if the question implies "top N" or similar.
"""


def _call_groq(messages: list[dict]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com/keys"
        )

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=30,
    )
    if response.status_code == 404:
        raise RuntimeError(
            f"Groq model '{GROQ_MODEL}' was not found (404). Groq periodically "
            "deprecates models. Check current model IDs at "
            "https://console.groq.com/docs/models or GET "
            "https://api.groq.com/openai/v1/models with your API key, then "
            "update GROQ_MODEL in app/llm.py."
        )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_prompt(question: str, retrieved_tables: list, few_shot_examples: list,
                  dialect: str, conversation_history: list[dict] | None = None) -> list[dict]:
    """
    Builds the message list sent to the LLM, using only RETRIEVED schema
    chunks (not the full DB) and RETRIEVED few-shot examples (not a static
    fixed set) - this is the RAG part of the pipeline.
    """
    schema_text = "\n\n".join(chunk.text for chunk, _score in retrieved_tables)

    examples_text = ""
    if few_shot_examples:
        example_blocks = [
            f"Q: {ex.question}\nSQL: {ex.sql}" for ex in few_shot_examples
        ]
        examples_text = "\n\nSimilar past examples:\n" + "\n\n".join(example_blocks)

    user_prompt = f"""Database dialect: {dialect}

Relevant tables (retrieved for this question):
{schema_text}
{examples_text}

Question: {question}

SQL query:"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_sql_with_retries(question: str, retrieved_tables: list, few_shot_examples: list,
                                dialect: str, execute_and_validate_fn,
                                conversation_history: list[dict] | None = None):
    """
    Generates SQL, then attempts to validate+execute it via
    execute_and_validate_fn(sql) -> (safe_sql, columns, rows, elapsed_ms).
    On failure, feeds the error back to the model and retries, up to
    MAX_RETRIES times. Returns (safe_sql, columns, rows, elapsed_ms, attempts, raw_sql_history).

    execute_and_validate_fn should raise an exception with a useful message
    on failure (validation error or DB execution error) - that message is
    what gets fed back to the model for self-correction.
    """
    messages = build_prompt(question, retrieved_tables, few_shot_examples, dialect, conversation_history)
    raw_sql_history = []
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):  # initial try + MAX_RETRIES retries
        raw_sql = _call_groq(messages)
        raw_sql_history.append(raw_sql)

        try:
            safe_sql, columns, rows, elapsed_ms = execute_and_validate_fn(raw_sql)
            return safe_sql, columns, rows, elapsed_ms, attempt, raw_sql_history
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw_sql})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That query failed with this error:\n{e}\n\n"
                        "Fix the SQL and output ONLY the corrected query, "
                        "no explanation."
                    ),
                })

    raise last_error
