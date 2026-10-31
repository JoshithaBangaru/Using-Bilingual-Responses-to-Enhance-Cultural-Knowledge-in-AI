"""Builds one fully filled evaluation-table record (pipeline + grading + logging)."""
import json
import re

import config
import database
import llm
from kb_retriever import search_kb
from pipeline import run_pipeline
from table_utils import CATEGORIES, detect_answer_type, to_table_row
from textutils import is_correct, to_answer_list

GRADER_PROMPT = """You are grading a cultural question-answering system.
Country: {country}
Question: {question}
Reference answer(s): {reference}
Final answer given by the system: {final}
Retrieved evidence (Wikipedia facts):
{evidence}

Return ONLY a JSON object with these keys:
"category": the best fit from {categories},
"correct": "yes", "no" or "unsure",
"kb_grounded": "yes", "no" or "na".

Rules:
- correct: if reference answers are given, "yes" only if the final answer names the same thing as one of
  them (spelling variants, translations and synonyms count) and does not contradict it, else "no".
  If no reference is given, judge from your own knowledge: "yes", "no" or "unsure".
- kb_grounded: "yes" only if the main claim of the final answer is stated in the retrieved evidence;
  "no" if evidence exists but does not support it; "na" if the evidence is "none"."""


def _json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(re.search(r"\{.*\}", cleaned, re.S).group(0))


def _grade(question, country, final, evidence, reference):
    ev = "\n".join(f"- {str(f.get('fact', ''))[:300]}" for f in evidence) if evidence else "none"
    prompt = GRADER_PROMPT.format(
        country=country or "unspecified", question=question,
        reference="; ".join(reference) if reference else "none given",
        final=final, evidence=ev, categories=CATEGORIES)
    try:
        data = _json(llm.generate(prompt, json_mode=True))
    except Exception:
        return {}
    category = data.get("category")
    return {"category": category if category in CATEGORIES else "Other",
            "correct": str(data.get("correct", "")).lower(),
            "kb_grounded": str(data.get("kb_grounded", "")).lower()}


def build_record(question, country, language=None, mode="dual_kb", answer_type=None, reference=None,
                 category=None, question_id=None, run_id="web", show_evidence=True,
                 assess_without_reference=True, local_question=None):
    """Run the pipeline for one question and return {'row': table row, 'timings': ..., 'result': ...}."""
    database.init_db()
    answer_type = answer_type or detect_answer_type(question)
    fmt = "mcq" if answer_type == "mcq" else "free"
    language = language or config.COUNTRY_LANGUAGE.get(country, "English")
    reference = to_answer_list(reference) if reference not in (None, "") else None

    result = run_pipeline(question, country, language, mode, fmt, local_question)

    evidence = result["retrieved_facts"]
    evidence_used = bool(evidence)
    if not evidence and show_evidence:                  # retrieved only to display / check grounding
        evidence = search_kb(question, country or None, config.TOP_K)
    scores = [f["score"] for f in evidence if "score" in f]
    relevance = result["evidence_relevance"] if evidence_used else (
        round(sum(scores) / len(scores), 4) if scores else None)

    final_en = result["final_answer_en"]
    need_llm = bool(evidence) or not category or (answer_type == "saq" and (reference or assess_without_reference))
    grade = _grade(question, country, final_en, evidence, reference) if need_llm else {}

    # ---- Final Correct (the basis is always recorded, so it can be audited)
    if reference and answer_type == "mcq":
        ok = is_correct(final_en, reference[0], "mcq")
        final_correct, basis = ("Yes" if ok else "No"), "Option letter match"
    elif reference:
        if is_correct(final_en, reference, "saq"):
            final_correct, basis = "Yes", "Reference match"
        elif grade.get("correct") == "yes":
            final_correct, basis = "Yes", "LLM vs reference"
        else:
            final_correct, basis = "No", "LLM vs reference" if grade else "Reference match"
    elif assess_without_reference and grade:
        verdict = grade.get("correct")
        final_correct = {"yes": "Yes", "no": "No"}.get(verdict, "Unsure")
        basis = "LLM-assessed (no reference)"
    else:
        final_correct, basis = "Not graded", "-"

    grounded_raw = grade.get("kb_grounded")
    if not evidence:
        kb_grounded = "N/A (no evidence)"
    else:
        kb_grounded = {"yes": "Yes", "no": "No"}.get(grounded_raw, "Unknown")

    category = category or grade.get("category") or "Unclassified"

    record = {
        "run_id": run_id, "mode": mode, "question_id": question_id, "country": country,
        "language": result["language"], "answer_type": answer_type, "question": question,
        "ground_truth": reference, "local_answer": result["local_answer"],
        "local_answer_en": result["local_answer_en"], "english_answer": result["english_answer"],
        "similarity": result["similarity_score"], "is_disagreed": result["is_disagreed"],
        "used_judge": result["used_judge"], "retrieved_facts": evidence,
        "final_answer": result["final_answer"],
        "is_correct": {"Yes": 1, "No": 0}.get(final_correct),
        "english_correct": int(is_correct(result["english_answer"], reference[0] if answer_type == "mcq" else reference, answer_type)) if reference else None,
        "local_correct": int(is_correct(result["local_answer_en"], reference[0] if answer_type == "mcq" else reference, answer_type)) if reference else None,
        "latency_ms": result["latency_ms"], "category": category, "agreement": result["agreement"],
        "judge_decision": result["judge_decision"], "evidence_relevance": relevance,
        "evidence_used": int(evidence_used), "kb_grounded": kb_grounded,
        "final_correct": final_correct, "correctness_basis": basis, "timings": result["timings"],
    }
    record["id"] = database.log_run(record)
    return {"row": to_table_row(record), "timings": result["timings"], "result": result}


def rows_for_run(run_id):
    """Rebuild table rows for a saved run (web or command line)."""
    database.init_db()
    out = []
    for rec in database.fetch_runs(run_id):
        if rec.get("error"):
            continue
        out.append(to_table_row(rec))
    return out
