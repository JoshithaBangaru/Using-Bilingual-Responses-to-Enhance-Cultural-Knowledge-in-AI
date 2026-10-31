"""Lightweight helpers for the evaluation table (safe to import from the Streamlit app:
no torch / FAISS / LLM imports)."""
import csv
import io
import json
import re
import statistics

import config
from textutils import to_answer_list

CATEGORIES = ["Food", "Festival/Holiday", "Clothing", "Music/Dance", "Sports/Leisure",
              "Family/Customs", "Education/Work", "Religion", "Other"]

# The proposal's evaluation record schema (plus Ground Truth and Correctness Basis,
# which make the Final Correct column auditable).
TABLE_COLUMNS = [
    "Record ID", "Country/Region", "Local Language", "Resource Level", "Cultural Category",
    "Question Type", "Question", "Ground Truth", "Local Answer", "English Answer",
    "Similarity", "Agreement", "Disagreement", "Retrieved Evidence", "Evidence Relevance",
    "Judge Decision", "Final Answer", "Final Correct", "Correctness Basis", "KB Grounded",
    "Latency (ms)",
]

REFERENCE_BASES = ("Option letter match", "Reference match", "LLM vs reference")


def detect_answer_type(question):
    has_options = re.search(r"(^|\n)\s*A[\.\)]\s", question) and re.search(r"\n\s*B[\.\)]\s", question)
    return "mcq" if has_options else "saq"


# ------------------------------------------------------------------ question input
def _item(question, country, reference=None, category=None, question_id=None, answer_type=None):
    ref = to_answer_list(reference) if reference not in (None, "") else None
    return {"question": question.strip(), "country": config.normalize_country(country),
            "reference": ref or None, "category": (category or "").strip() or None,
            "question_id": question_id, "answer_type": answer_type or detect_answer_type(question)}


def parse_questions_text(text, default_country):
    """One question per line: 'question | reference | category' or 'Country | question | reference | category'."""
    items = []
    for i, line in enumerate(l for l in text.splitlines() if l.strip()):
        parts = [p.strip() for p in line.split("|")]
        country = default_country
        if config.normalize_country(parts[0]) in config.COUNTRIES and len(parts) > 1:
            country = parts.pop(0)
        question = parts[0].replace("\\n", "\n")          # write \n for a line break (MCQ options)
        reference = parts[1] if len(parts) > 1 else None
        category = parts[2] if len(parts) > 2 else None
        items.append(_item(question, country, reference, category, f"q{i + 1}"))
    return items


def parse_upload(filename, raw_bytes, default_country):
    """CSV (columns: question, country, ground_truth/reference/answer, category) or JSONL or TXT."""
    text = raw_bytes.decode("utf-8-sig")
    name = filename.lower()
    rows = []
    if name.endswith(".jsonl") or name.endswith(".json"):
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    elif name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
    else:
        return parse_questions_text(text, default_country)
    items = []
    for i, row in enumerate(rows):
        row = {str(k).strip().lower(): v for k, v in row.items()}
        question = row.get("question") or row.get("question_en") or ""
        if not str(question).strip():
            continue
        reference = row.get("ground_truth") or row.get("reference") or row.get("answer") or row.get("expected")
        items.append(_item(str(question), row.get("country") or default_country, reference,
                           row.get("category"), str(row.get("question_id") or f"q{i + 1}"),
                           row.get("answer_type")))
    return items


# ------------------------------------------------------------------ table rows
def _facts(rec):
    facts = rec.get("retrieved_facts")
    if isinstance(facts, str):
        try:
            facts = json.loads(facts)
        except ValueError:
            facts = []
    return facts or []


def evidence_text(facts, used):
    if not facts:
        return "-"
    lines = [f"[{f.get('topic', '')}] {str(f.get('fact', ''))[:220]}" for f in facts if isinstance(f, dict)]
    body = "\n".join(lines)
    return body if used else "(shown for reference; not used in the decision)\n" + body


def to_table_row(rec):
    """Build one table row from a DB-style record dict."""
    sim = rec.get("similarity")
    dis = rec.get("is_disagreed")
    truth = to_answer_list(rec.get("ground_truth"))
    rel = rec.get("evidence_relevance")
    return {
        "Record ID": rec.get("id"),
        "Country/Region": rec.get("country"),
        "Local Language": rec.get("language"),
        "Resource Level": config.RESOURCE_LEVEL.get(rec.get("country"), "unknown"),
        "Cultural Category": rec.get("category") or "Unclassified",
        "Question Type": "MCQ" if rec.get("answer_type") == "mcq" else "Short answer",
        "Question": rec.get("question"),
        "Ground Truth": ", ".join(truth) if truth else "-",
        "Local Answer": rec.get("local_answer"),
        "English Answer": rec.get("english_answer"),
        "Similarity": round(float(sim), 4) if sim is not None else None,
        "Agreement": rec.get("agreement") or "N/A",
        "Disagreement": "N/A" if dis is None else ("Yes" if dis else "No"),
        "Retrieved Evidence": evidence_text(_facts(rec), bool(rec.get("evidence_used"))),
        "Evidence Relevance": round(float(rel), 3) if rel is not None else None,
        "Judge Decision": rec.get("judge_decision") or "N/A",
        "Final Answer": rec.get("final_answer"),
        "Final Correct": rec.get("final_correct") or "Not graded",
        "Correctness Basis": rec.get("correctness_basis") or "-",
        "KB Grounded": rec.get("kb_grounded") or "N/A",
        "Latency (ms)": round(float(rec["latency_ms"]), 1) if rec.get("latency_ms") is not None else None,
    }


# ------------------------------------------------------------------ summary
def _rate(rows):
    return (sum(r["Final Correct"] == "Yes" for r in rows) / len(rows) * 100) if rows else None


def summarize(rows):
    """Headline numbers. Reference-graded accuracy is the trustworthy one."""
    ref = [r for r in rows if r["Correctness Basis"] in REFERENCE_BASES]
    llm = [r for r in rows if str(r["Correctness Basis"]).startswith("LLM-assessed")]
    disagreed = [r for r in rows if r["Disagreement"] == "Yes"]
    grounded_pool = [r for r in rows if r["KB Grounded"] in ("Yes", "No")]
    latencies = [r["Latency (ms)"] for r in rows if r["Latency (ms)"] is not None]
    return {
        "records": len(rows),
        "accuracy_reference": _rate(ref), "n_reference": len(ref),
        "accuracy_llm_assessed": _rate(llm), "n_llm_assessed": len(llm),
        "disagreement_rate": (len(disagreed) / len(rows) * 100) if rows else None,
        "kb_grounded_rate": (sum(r["KB Grounded"] == "Yes" for r in grounded_pool) / len(grounded_pool) * 100)
        if grounded_pool else None,
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_mean_ms": statistics.mean(latencies) if latencies else None,
    }


def accuracy_by(rows, key):
    """Accuracy (reference-graded rows only) grouped by a table column, e.g. 'Country/Region'."""
    groups = {}
    for r in rows:
        if r["Correctness Basis"] in REFERENCE_BASES:
            groups.setdefault(r[key], []).append(r)
    return {k: {"n": len(v), "accuracy_%": round(_rate(v), 1)} for k, v in sorted(groups.items())}


def mean_timings(timing_dicts):
    keys = ["generation_ms", "similarity_ms", "retrieval_ms", "judge_ms"]
    dicts = [t for t in timing_dicts if t]
    if not dicts:
        return {}
    return {k: round(sum(t.get(k, 0) for t in dicts) / len(dicts), 1) for k in keys}
