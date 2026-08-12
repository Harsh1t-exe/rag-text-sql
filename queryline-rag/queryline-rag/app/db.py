"""
Connects to a target database, introspects its schema as individual
per-table "chunks" (so they can be embedded/retrieved independently by the
RAG layer), and executes validated read-only queries safely.
"""
import time
from dataclasses import dataclass
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

QUERY_TIMEOUT_SECONDS = 15
MAX_ROWS_HARD_CAP = 500


@dataclass
class TableChunk:
    """One table's schema, formatted as a retrievable RAG document."""
    table_name: str
    text: str          # the chunk fed to the retriever / prompt
    column_names: list


def get_engine(connection_string: str) -> Engine:
    """
    Creates a SQLAlchemy engine for the given connection string.
    Works with sqlite:///, postgresql+psycopg2://, mysql+pymysql://, etc.
    For production use against real user databases, the connection string's
    role should be granted SELECT only - enforce this at the DB level.
    """
    return create_engine(
        connection_string,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 10} if "sqlite" not in connection_string else {},
    )


def get_table_chunks(engine: Engine, max_tables: int = 200) -> list[TableChunk]:
    """
    Introspects the database and returns one TableChunk per table.
    Each chunk is a self-contained text blob: table name, columns with
    types, and any foreign keys - exactly what a retriever needs to decide
    "is this table relevant to the question?" independent of every other
    table in the database.
    """
    inspector = inspect(engine)
    table_names = inspector.get_table_names()[:max_tables]

    chunks = []
    for table in table_names:
        columns = inspector.get_columns(table)
        col_names = [c["name"] for c in columns]
        col_descs = [f"{c['name']} {c['type']}" for c in columns]

        fk_lines = []
        for fk in inspector.get_foreign_keys(table):
            if fk.get("constrained_columns") and fk.get("referred_table"):
                fk_lines.append(
                    f"{table}.{fk['constrained_columns'][0]} -> "
                    f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                )

        text_parts = [f"Table {table}({', '.join(col_descs)})"]
        for fk in fk_lines:
            text_parts.append(f"  -- FK: {fk}")

        chunks.append(TableChunk(
            table_name=table,
            text="\n".join(text_parts),
            column_names=col_names,
        ))

    return chunks


def run_readonly_query(engine: Engine, sql: str, timeout_seconds: int = QUERY_TIMEOUT_SECONDS):
    """
    Executes an already-validated SELECT query with a wall-clock timeout
    guard and a hard row cap, returns (columns, rows, elapsed_ms).
    """
    start = time.time()
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(text(sql))
        columns = list(result.keys())
        rows = []
        for i, row in enumerate(result):
            rows.append(list(row))
            if i >= MAX_ROWS_HARD_CAP:
                break
            if time.time() - start > timeout_seconds:
                raise TimeoutError("Query exceeded time limit and was stopped.")
    elapsed_ms = int((time.time() - start) * 1000)
    return columns, rows, elapsed_ms
