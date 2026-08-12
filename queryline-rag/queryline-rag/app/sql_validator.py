"""
Validates LLM-generated SQL before it's ever executed.
Rule: only single, read-only SELECT statements are allowed. Everything else
is rejected. This is the most important file in the project from a safety
standpoint - do not weaken these checks.
"""
import sqlglot
from sqlglot import exp

# Built defensively with getattr: not every sqlglot version ships every
# expression class (e.g. Grant/TruncateTable have moved/been renamed across
# releases). Any class missing on the installed version is simply skipped -
# the keyword-level check below is the backstop for those cases.
_CANDIDATE_FORBIDDEN_NAMES = (
    "Insert", "Update", "Delete", "Drop", "Alter",
    "Create", "TruncateTable", "Merge", "Grant",
)
FORBIDDEN_STATEMENT_TYPES = tuple(
    getattr(exp, name) for name in _CANDIDATE_FORBIDDEN_NAMES if hasattr(exp, name)
)


class UnsafeSQLError(Exception):
    pass


def clean_sql(raw_sql: str) -> str:
    """Strip markdown code fences etc. that LLMs love to add."""
    sql = raw_sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        sql = "\n".join(lines).strip()
    if sql.lower().startswith("sql\n"):
        sql = sql[4:].strip()
    return sql.rstrip(";").strip()


def validate_select_only(raw_sql: str, dialect: str = "postgres") -> str:
    """
    Returns the cleaned, validated SQL string if safe.
    Raises UnsafeSQLError if the query is anything other than a single
    read-only SELECT.
    """
    sql = clean_sql(raw_sql)
    if not sql:
        raise UnsafeSQLError("Model returned an empty query.")

    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as e:
        raise UnsafeSQLError(f"Could not parse generated SQL: {e}")

    statements = [s for s in statements if s is not None]

    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Expected exactly one SQL statement, found {len(statements)}. "
            "Multi-statement queries are not allowed."
        )

    stmt = statements[0]

    if not isinstance(stmt, exp.Select):
        raise UnsafeSQLError(
            f"Only SELECT statements are allowed. Got: {type(stmt).__name__}"
        )

    for forbidden_type in FORBIDDEN_STATEMENT_TYPES:
        if list(stmt.find_all(forbidden_type)):
            raise UnsafeSQLError(
                f"Query contains a forbidden operation: {forbidden_type.__name__}"
            )

    banned_keywords = {
        "insert", "update", "delete", "drop", "alter", "create",
        "truncate", "grant", "revoke", "attach", "pragma", "exec", "execute",
    }
    lowered = sql.lower()
    for kw in banned_keywords:
        if f" {kw} " in f" {lowered} " or lowered.startswith(kw):
            raise UnsafeSQLError(f"Query contains banned keyword: {kw}")

    return sql


def enforce_row_limit(sql: str, max_rows: int = 500, dialect: str = "postgres") -> str:
    """Adds/lowers a LIMIT clause so no query can return unbounded rows."""
    parsed = sqlglot.parse_one(sql, read=dialect)
    existing_limit = parsed.args.get("limit")

    if existing_limit is not None:
        try:
            current = int(existing_limit.expression.this)
            if current > max_rows:
                parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (AttributeError, ValueError, TypeError):
            parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return parsed.sql(dialect=dialect)
