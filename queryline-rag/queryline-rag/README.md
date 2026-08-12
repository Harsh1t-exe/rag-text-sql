# Queryline RAG — Retrieval-Augmented Text-to-SQL

Connect any database, ask it questions in plain English. Unlike a naive
text-to-SQL wrapper that stuffs the entire schema into every prompt, this
version **retrieves** only the relevant tables and similar past examples
per question, and **self-corrects** when generated SQL fails to execute.

## Why RAG matters here (the pitch)

A naive text-to-SQL tool dumps the whole schema into the prompt every time.
That works for a toy 4-table demo. It breaks down on real databases:
- Large schemas blow past context limits or drown the model in irrelevant tables.
- The model is more likely to hallucinate joins through tables it shouldn't touch.
- Prompt cost scales with schema size even when the question only needs 2 tables.

**Schema-retrieval RAG** (aka "schema linking" in the text-to-SQL literature)
solves this: index each table as a separate retrievable chunk, and at query
time, only pull in the tables that are actually relevant to the question.
This project also does **few-shot retrieval** — pulling similar past
(question, SQL) pairs into the prompt, which measurably improves accuracy on
schema-specific conventions an LLM can't infer from column names alone.

## Architecture

```
 connect DB ──▶ introspect schema into per-table chunks
                          │
                          ▼
              build TF-IDF index over chunks (SchemaRetriever)
                          │
                     [session created]
                          │
question ──▶ SchemaRetriever.retrieve(question, top_k=5)
              │
              ├──▶ FewShotRetriever.retrieve(question, top_k=3)
              │
              ▼
        prompt = retrieved tables + retrieved examples + question
              │
              ▼
        LLM (Groq / Llama 3.3 70B) generates SQL
              │
              ▼
        sqlglot validation (SELECT-only, single statement, row-limited)
              │
        ┌─────┴─────┐
     fails          succeeds
        │               │
        ▼               ▼
  feed error back    execute against DB
  to LLM, retry      (read-only role)
  (up to 2x)              │
                           ▼
                    results + retrieval trace
                    returned to user
```

## Why TF-IDF instead of embedding models

Schema text (table/column names) and short natural-language questions are
keyword-heavy, low-vocabulary text — TF-IDF + cosine similarity performs
well here, requires no model download, no GPU, and is instant and free.

**If you want to extend this for a resume talking point about embeddings**,
swap `app/schema_rag.py`'s `TfidfVectorizer` for `sentence-transformers`
(e.g. `all-MiniLM-L6-v2`) and store vectors in a proper vector DB (Chroma,
pgvector, or FAISS) instead of the in-memory scikit-learn matrix. The
retrieval interface (`SchemaRetriever.retrieve()`) is designed so that swap
doesn't touch anything else in the pipeline — that separation is itself
worth mentioning in an interview.

## Project structure

```
queryline-rag/
├── app/
│   ├── main.py            FastAPI app: /api/connect, /api/query, /api/feedback
│   ├── db.py                schema introspection into per-table chunks, query execution
│   ├── schema_rag.py         SchemaRetriever + FewShotRetriever (the RAG layer)
│   ├── llm.py                Groq API calls, prompt construction, self-correction loop
│   ├── sessions.py            in-memory session store (one per connected DB)
│   ├── sql_validator.py       blocks unsafe SQL before it ever runs
│   └── seed_examples.json     starter few-shot bank for the sample DB
├── sample_data/
│   └── create_sample_db.py   10-table demo DB (store + support + marketing)
├── eval/
│   ├── eval_set.json          25 question/expected-SQL pairs
│   └── run_eval.py            measures execution-match accuracy against the eval set
├── static/index.html          frontend (shows retrieval trace + self-correction attempts live)
├── requirements.txt
├── Dockerfile
└── render.yaml
```

## Safety model

Same non-negotiables as any text-to-SQL system exposed to a real database:
- The LLM only ever sees table/column names and types — never data, never credentials.
- Generated SQL is parsed with `sqlglot` and rejected unless it's exactly one `SELECT` statement.
- Hard row limit (500) and query timeout on every execution.
- **Always connect with a database role that only has `SELECT` granted.** App-level validation is a second line of defense, not the first.

```sql
-- Postgres example: create this role before pointing Queryline at a real DB
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE yourdb TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;
```

## Run it locally

```bash
cd queryline-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python sample_data/create_sample_db.py
```

Get a free Groq key (no card needed) at https://console.groq.com/keys:

```bash
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --reload
```

Open http://localhost:8000. Click **Connect** (leave the field blank to use
the bundled 10-table sample DB), then try a question. Watch the "Retrieved
tables" trace under the result — that's the RAG step working, visibly.

## Run the eval

With the server running and `GROQ_API_KEY` set:

```bash
python eval/run_eval.py
```

This measures **execution accuracy** (does the generated SQL return the
same result set as hand-written correct SQL) — the standard metric used in
text-to-SQL research benchmarks like Spider. It also reports average
self-correction rate (how often the first attempt needed a retry) and
latency. Put these numbers in your README/resume once you've run it —
"87% execution accuracy on a 25-question eval set" is a concrete, defensible
claim that a plain demo can't make.

## Deploy for free (Render)

1. Push to GitHub.
2. Render → New + → Web Service → connect your repo.
3. **If your files are nested in a subfolder** (e.g. `your-repo/queryline-rag/...`), set **Root Directory** to that folder name in Settings — this is the single most common deploy failure.
4. Build command: `pip install -r requirements.txt && python sample_data/create_sample_db.py`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` — note `$PORT`, not a hardcoded port; Render assigns it dynamically.
6. Add environment variable `GROQ_API_KEY`.
7. If you hit Python-version-related dependency errors, pin `PYTHON_VERSION=3.11.9` as an env var — some packages (like `sqlglot`) can lag behind the newest Python release Render defaults to.

## Multi-turn follow-ups

Sessions retain conversation history, so you can ask a follow-up like
*"only show the cancelled ones"* after a broader question, and it'll be
interpreted in context (last 3 turns are included in the prompt).

## What to say about this project in an interview

- **RAG for structured data, not just documents** — most RAG demos are
  "chat with a PDF." This applies retrieval to schema linking, a genuinely
  different and less commonly-demoed application of the pattern.
- **Self-correction loop** — the system observes execution failures and
  retries with the error in context, rather than being purely one-shot.
- **Measured, not just claimed** — the eval script gives you a real accuracy
  number instead of "it seems to work."
- **Safety-first execution** — be ready to walk through the sqlglot
  validation layer; it's the most interesting engineering decision in the
  codebase and shows you think about AI-generated code as an attack surface.

## Roadmap ideas

- Swap TF-IDF for real embeddings (sentence-transformers + pgvector) once you want a second retrieval method to compare against.
- Persist sessions and few-shot banks to Postgres/Redis instead of in-memory, so they survive restarts and scale across processes.
- Add a "confidence" signal — e.g. flag queries where retrieval scores were low, meaning the model may be working from a poor schema match.
- Stream SQL generation token-by-token for a snappier UX.
