"""
Queryline RAG - FastAPI backend.

Flow:
  POST /api/connect  -> introspects a DB, builds schema RAG index,
                         returns a session_id
  POST /api/query    -> retrieves relevant tables + similar few-shot
                         examples for the question, generates SQL,
                         self-corrects on failure (up to 2 retries),
                         executes, returns results + retrieval trace
  POST /api/feedback  -> marks a query as correct, adding it to the
                         few-shot bank for that session (RAG grows over time)
  GET  /api/health
"""
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.db import get_engine, get_table_chunks, run_readonly_query
from app.llm import generate_sql_with_retries
from app.schema_rag import SchemaRetriever, FewShotExample
from app.sql_validator import validate_select_only, enforce_row_limit, UnsafeSQLError
from app.sessions import session_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("queryline")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "sample_data" / "sample_store.db"
DEFAULT_CONNECTION_STRING = f"sqlite:///{DEFAULT_DB_PATH}"
SEED_EXAMPLES_PATH = Path(__file__).resolve().parent / "seed_examples.json"

app = FastAPI(title="Queryline RAG - Text-to-SQL")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def dialect_from_connection_string(cs: str) -> str:
    if cs.startswith("postgresql"):
        return "postgres"
    if cs.startswith("mysql"):
        return "mysql"
    if cs.startswith("sqlite"):
        return "sqlite"
    return "postgres"


def load_seed_examples() -> list[FewShotExample]:
    if not SEED_EXAMPLES_PATH.exists():
        return []
    data = json.loads(SEED_EXAMPLES_PATH.read_text())
    return [FewShotExample(question=d["question"], sql=d["sql"]) for d in data]


class ConnectRequest(BaseModel):
    connection_string: str | None = None  # None => bundled sample DB


class ConnectResponse(BaseModel):
    session_id: str
    table_count: int
    tables: list[str]
    used_sample_db: bool


class QueryRequest(BaseModel):
    session_id: str
    question: str


class RetrievedTable(BaseModel):
    table_name: str
    relevance_score: float


class QueryResponse(BaseModel):
    sql: str
    columns: list[str]
    rows: list[list]
    row_count: int
    elapsed_ms: int
    attempts: int
    retrieved_tables: list[RetrievedTable]
    few_shot_used: int


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    sql: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest):
    connection_string = req.connection_string or DEFAULT_CONNECTION_STRING
    used_sample_db = req.connection_string is None

    try:
        engine = get_engine(connection_string)
        chunks = get_table_chunks(engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not connect / read schema: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No tables found in this database.")

    dialect = dialect_from_connection_string(connection_string)
    retriever = SchemaRetriever(chunks)

    seed = load_seed_examples() if used_sample_db else []
    session = session_store.create(
        engine=engine, dialect=dialect, schema_retriever=retriever,
        table_count=len(chunks), seed_examples=seed,
    )

    return ConnectResponse(
        session_id=session.session_id,
        table_count=len(chunks),
        tables=[c.table_name for c in chunks],
        used_sample_db=used_sample_db,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    session = session_store.get(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Call /api/connect again.",
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retrieved = session.schema_retriever.retrieve(req.question, top_k=5)
    if not retrieved:
        raise HTTPException(status_code=400, detail="No tables available to query against.")

    few_shot = session.few_shot_retriever.retrieve(req.question, top_k=3)

    def execute_and_validate(raw_sql: str):
        safe_sql = validate_select_only(raw_sql, dialect=session.dialect)
        safe_sql = enforce_row_limit(safe_sql, max_rows=500, dialect=session.dialect)
        columns, rows, elapsed_ms = run_readonly_query(session.engine, safe_sql)
        return safe_sql, columns, rows, elapsed_ms

    try:
        safe_sql, columns, rows, elapsed_ms, attempts, raw_history = generate_sql_with_retries(
            question=req.question,
            retrieved_tables=retrieved,
            few_shot_examples=few_shot,
            dialect=session.dialect,
            execute_and_validate_fn=execute_and_validate,
            conversation_history=session.conversation_history[-6:],  # last 3 turns
        )
    except (UnsafeSQLError, TimeoutError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed after retries: {e}")

    # update multi-turn conversation memory
    session.conversation_history.append({"role": "user", "content": req.question})
    session.conversation_history.append({"role": "assistant", "content": safe_sql})

    return QueryResponse(
        sql=safe_sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        elapsed_ms=elapsed_ms,
        attempts=attempts,
        retrieved_tables=[
            RetrievedTable(table_name=c.table_name, relevance_score=round(score, 3))
            for c, score in retrieved
        ],
        few_shot_used=len(few_shot),
    )


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    session = session_store.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    session.few_shot_retriever.add_example(req.question, req.sql)
    return {"status": "added", "bank_size": len(session.few_shot_retriever.examples)}


static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(static_dir / "index.html"))
