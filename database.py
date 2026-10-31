"""One SQLite table that stores every API request and every evaluation row."""
import json
import sqlite3

import config

COLUMNS = {
    "run_id": "TEXT", "mode": "TEXT", "question_id": "TEXT", "country": "TEXT",
    "language": "TEXT", "answer_type": "TEXT", "question": "TEXT", "ground_truth": "TEXT",
    "local_answer": "TEXT", "local_answer_en": "TEXT", "english_answer": "TEXT",
    "similarity": "REAL", "is_disagreed": "INTEGER", "used_judge": "INTEGER",
    "retrieved_facts": "TEXT", "final_answer": "TEXT", "is_correct": "INTEGER",
    "english_correct": "INTEGER", "local_correct": "INTEGER", "error": "TEXT",
    "latency_ms": "REAL",
    # evaluation-table columns
    "category": "TEXT", "agreement": "TEXT", "judge_decision": "TEXT",
    "evidence_relevance": "REAL", "evidence_used": "INTEGER", "kb_grounded": "TEXT",
    "final_correct": "TEXT", "correctness_basis": "TEXT", "timings": "TEXT",
}


def _conn():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    cols = ", ".join(f"{name} {typ}" for name, typ in COLUMNS.items())
    conn = _conn()
    try:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS evaluation_records ("
            f"id INTEGER PRIMARY KEY AUTOINCREMENT, {cols}, "
            f"created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        # migrate databases created by an older version of this file
        existing = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_records)")}
        for name, typ in COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE evaluation_records ADD COLUMN {name} {typ}")
        conn.commit()
    finally:
        conn.close()


def _clean(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def log_run(record: dict) -> int:
    """Insert one row and return its Record ID."""
    names = list(COLUMNS)
    conn = _conn()
    try:
        cursor = conn.execute(
            f"INSERT INTO evaluation_records ({', '.join(names)}) "
            f"VALUES ({', '.join('?' for _ in names)})",
            [_clean(record.get(n)) for n in names],
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def already_logged(run_id, question_id, mode) -> bool:
    """True if a SUCCESSFUL row exists (errored rows are retried on re-run)."""
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM evaluation_records WHERE run_id=? AND question_id=? AND mode=? "
            "AND (error IS NULL OR error='') LIMIT 1", (run_id, question_id, mode)).fetchone()
    finally:
        conn.close()
    return row is not None


def delete_errors(run_id, question_id, mode):
    conn = _conn()
    try:
        conn.execute(
            "DELETE FROM evaluation_records WHERE run_id=? AND question_id=? AND mode=? "
            "AND error IS NOT NULL AND error<>''", (run_id, question_id, mode))
        conn.commit()
    finally:
        conn.close()


def fetch_runs(run_id=None):
    conn = _conn()
    try:
        if run_id:
            rows = conn.execute("SELECT * FROM evaluation_records WHERE run_id=? ORDER BY id",
                                (run_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM evaluation_records ORDER BY id").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def list_run_ids():
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT run_id, MAX(id) m FROM evaluation_records GROUP BY run_id ORDER BY m DESC"
        ).fetchall()
    finally:
        conn.close()
    return [r["run_id"] for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"SQLite database initialised at {config.DB_PATH}")
