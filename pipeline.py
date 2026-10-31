"""The full dual-language pipeline, usable from the API, the web app and the evaluation scripts."""
import time

import config
import judge
from disagreement import check_disagreement
from dual_gen import generate_bilingual_answers
from kb_retriever import search_kb
from textutils import extract_letter


def _ms(since):
    return round((time.perf_counter() - since) * 1000, 1)


def agreement_label(similarity, is_disagreed, answer_format="free", threshold=None):
    """Agree / Partial / Disagree (the proposal's 'Agreement' column)."""
    threshold = config.DISAGREEMENT_THRESHOLD if threshold is None else threshold
    if is_disagreed is None:
        return "N/A"
    if answer_format == "mcq":
        return "Disagree" if is_disagreed else "Agree"
    if similarity is None:
        return "N/A"
    if similarity >= threshold:
        return "Agree"
    return "Partial" if similarity >= threshold - 0.10 else "Disagree"


def run_pipeline(question: str, country: str = "", language: str = "English",
                 mode: str = "dual_kb", answer_format: str = "free",
                 local_question: str = None, top_k: int = None, threshold: float = None) -> dict:
    """Run one question through the chosen pipeline variant (see config.MODES).

    english_only / local_only : baselines, no comparison, no judge
    dual_judge_nokb           : judge on disagreement, WITHOUT KB facts (ablation)
    dual_kb                   : judge on disagreement, WITH KB facts (main system)
    dual_kb_always            : judge with KB facts on every question (ablation)
    """
    if mode not in config.MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {config.MODES}")
    top_k = top_k or config.TOP_K
    threshold = config.DISAGREEMENT_THRESHOLD if threshold is None else threshold
    total0 = time.perf_counter()
    timings = {"generation_ms": 0.0, "similarity_ms": 0.0, "retrieval_ms": 0.0, "judge_ms": 0.0}

    t = time.perf_counter()
    gen = generate_bilingual_answers(question, country, language,
                                     local_question=local_question, answer_format=answer_format)
    timings["generation_ms"] = _ms(t)
    local_ans, local_en, english_ans = gen["local_answer"], gen["local_answer_en"], gen["english_answer"]

    similarity, is_disagreed = None, None
    retrieved, used_judge = [], False
    decision = "N/A (baseline mode)"

    if mode == "english_only":
        final, final_en = english_ans, english_ans
    elif mode == "local_only":
        final, final_en = local_ans, local_en
    else:
        t = time.perf_counter()
        similarity, is_disagreed = check_disagreement(local_ans, english_ans, threshold)
        if answer_format == "mcq":     # for multiple choice, different letters = disagreement
            local_letter = extract_letter(local_en) or extract_letter(local_ans)
            english_letter = extract_letter(english_ans)
            if local_letter and english_letter:
                is_disagreed = local_letter != english_letter
        timings["similarity_ms"] = _ms(t)

        use_kb = mode in ("dual_kb", "dual_kb_always")
        if bool(is_disagreed) or mode == "dual_kb_always":
            if use_kb:
                t = time.perf_counter()
                retrieved = search_kb(query=question, country=country or None, top_k=top_k)
                timings["retrieval_ms"] = _ms(t)
            prompt = judge.build_judge_prompt(
                question=question, local_ans=local_ans, english_ans=english_ans,
                facts=retrieved, answer_format=answer_format, use_facts=use_kb,
                local_ans_en=local_en)
            t = time.perf_counter()
            final, decision = judge.run_judge_full(prompt)
            timings["judge_ms"] = _ms(t)
            used_judge = True
        else:
            final, decision = english_ans, "Not invoked (answers agree)"
        final_en = final

    scores = [f["score"] for f in retrieved if isinstance(f, dict) and "score" in f]
    return {
        "question": question, "country": country, "language": gen["local_language"],
        "mode": mode, "local_answer": local_ans, "local_answer_en": local_en,
        "english_answer": english_ans, "similarity_score": similarity,
        "is_disagreed": is_disagreed,
        "agreement": agreement_label(similarity, is_disagreed, answer_format, threshold),
        "used_judge": used_judge, "judge_decision": decision,
        "final_answer": final, "final_answer_en": final_en,
        "retrieved_facts": retrieved,
        "evidence_relevance": round(sum(scores) / len(scores), 4) if scores else None,
        "timings": timings,
        "latency_ms": _ms(total0),
    }
