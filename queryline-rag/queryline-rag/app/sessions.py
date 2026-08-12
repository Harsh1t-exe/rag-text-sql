"""
In-memory session store. A "session" is created once per connected database
(via /api/connect) and holds:
  - the SQLAlchemy engine
  - the schema retriever (RAG index over tables)
  - the few-shot retriever (RAG index over accumulated examples)
  - conversation history, for multi-turn follow-up questions

NOTE: this is intentionally in-memory and single-process, which is fine for
a demo/portfolio deployment. For real multi-user production use, swap this
for Redis (session state) + a persistent vector store (pgvector/Chroma) so
sessions survive restarts and work across multiple server processes.
"""
import uuid
import time
from dataclasses import dataclass, field

from sqlalchemy.engine import Engine
from app.schema_rag import SchemaRetriever, FewShotRetriever, FewShotExample

SESSION_TTL_SECONDS = 60 * 60 * 4  # 4 hours


@dataclass
class Session:
    session_id: str
    engine: Engine
    dialect: str
    schema_retriever: SchemaRetriever
    few_shot_retriever: FewShotRetriever
    conversation_history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    table_count: int = 0


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, engine: Engine, dialect: str, schema_retriever: SchemaRetriever,
               table_count: int, seed_examples: list[FewShotExample] | None = None) -> Session:
        self._evict_expired()
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            engine=engine,
            dialect=dialect,
            schema_retriever=schema_retriever,
            few_shot_retriever=FewShotRetriever(seed_examples or []),
            table_count=table_count,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        self._evict_expired()
        session = self._sessions.get(session_id)
        if session:
            session.last_used_at = time.time()
        return session

    def _evict_expired(self):
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used_at > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]


# Single process-wide store
session_store = SessionStore()
