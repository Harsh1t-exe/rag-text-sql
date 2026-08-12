"""
Runs the eval set against the running Queryline server and reports
execution-match accuracy: for each question, does the generated SQL
return the same result set as the hand-written expected SQL?

This is "execution accuracy" - the standard metric in text-to-SQL research
(used in benchmarks like Spider) - and is more meaningful than comparing
SQL strings directly, since two different queries can be equally correct.

Usage:
    1. Start the server: uvicorn app.main:app --reload
    2. Set GROQ_API_KEY
    3. Run: python eval/run_eval.py
"""
import json
import time
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"


def get_expected_result(sql: str, session_id: str):
    """Runs the hand-written expected SQL directly via a raw connection for comparison."""
    import sqlite3
    db_path = Path(__file__).resolve().parent.parent / "sample_data" / "sample_store.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return set(tuple(row) for row in rows)


def main():
    eval_cases = json.loads(EVAL_SET_PATH.read_text())

    resp = requests.post(f"{BASE_URL}/api/connect", json={})
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    print(f"Connected. Session: {session_id}\n")

    correct = 0
    failed_cases = []
    total_latency_ms = 0
    total_attempts = 0

    for i, case in enumerate(eval_cases, 1):
        question = case["question"]
        expected_sql = case["expected_sql"]

        try:
            expected_rows = get_expected_result(expected_sql, session_id)
        except Exception as e:
            print(f"[{i}] SKIP (bad expected SQL): {question} -> {e}")
            continue

        try:
            r = requests.post(f"{BASE_URL}/api/query", json={
                "session_id": session_id, "question": question
            }, timeout=60)
            if r.status_code != 200:
                failed_cases.append((question, f"HTTP {r.status_code}: {r.json().get('detail')}"))
                print(f"[{i}] ✗ {question}\n     ERROR: {r.json().get('detail')}")
                continue

            data = r.json()
            actual_rows = set(tuple(row) for row in data["rows"])
            total_latency_ms += data["elapsed_ms"]
            total_attempts += data["attempts"]

            if actual_rows == expected_rows:
                correct += 1
                print(f"[{i}] ✓ {question}  ({data['attempts']} attempt(s), {data['elapsed_ms']}ms)")
            else:
                failed_cases.append((question, f"Generated: {data['sql']}\n     Expected rows: {expected_rows}\n     Got rows: {actual_rows}"))
                print(f"[{i}] ✗ {question}\n     Generated SQL: {data['sql']}")

        except Exception as e:
            failed_cases.append((question, str(e)))
            print(f"[{i}] ✗ {question}\n     EXCEPTION: {e}")

        time.sleep(0.3)  # be gentle on Groq's free tier rate limit

    total = len(eval_cases)
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    if correct > 0:
        print(f"Avg latency: {total_latency_ms/total:.0f}ms")
        print(f"Avg attempts (self-correction rate): {total_attempts/total:.2f}")
    print(f"{'='*50}")

    if failed_cases:
        print("\nFailed cases:")
        for q, reason in failed_cases:
            print(f"  - {q}\n    {reason}")


if __name__ == "__main__":
    main()
