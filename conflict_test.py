"""Controlled conflict cases (proposal: 'Answer Perturbation / Conflict Cases').

For each short-answer question we BUILD a disagreement on purpose: one language holds the
correct answer, the other a plausible wrong one (both placements are tested to rule out
position bias). Then we measure:
  * detection rate : does the similarity check flag the pair as a disagreement?
  * judge accuracy : does the judge pick the correct answer WITHOUT KB facts vs WITH KB facts?
  * KB coverage    : do the retrieved facts even contain the correct answer?

    python conflict_test.py --input eval_data/curated_questions.jsonl --run-id conflict1
"""
import argparse
import csv
import json
import os
import re

import pandas as pd

import config
import data_loader
import judge
import llm
from disagreement import check_disagreement
from kb_retriever import search_kb
from textutils import is_correct_saq, to_answer_list

PROMPT = """You are helping build a test set for a cultural question-answering system.
Question about {country}: {question}
Correct answer: {gold}

Return ONLY a JSON object with these keys:
"correct_local": one short sentence in {language} saying the correct answer,
"wrong_short_en": a plausible but definitely INCORRECT answer in English (1-4 words; the same kind of thing, e.g. another real dish/festival/item from the same country),
"wrong_local": one short sentence in {language} saying that incorrect answer."""


def _json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    return json.loads(match.group(0))


def judged_correct(output, gold, wrong):
    """Right only if the gold answer is named and the injected wrong answer is NOT."""
    return is_correct_saq(output, gold) and not is_correct_saq(output, [wrong])


def run(samples, run_id, out_dir):
    rows = []
    for s in samples:
        if s["answer_type"] != "saq":
            continue
        gold = to_answer_list(s["ground_truth"])
        try:
            gen = _json(llm.generate(
                PROMPT.format(country=s["country"], question=s["question"], gold=gold[0],
                              language=s["language"]), json_mode=True))
            wrong = gen["wrong_short_en"].strip()
            correct_en = f"The answer is {gold[0]}."
            wrong_en = f"The answer is {wrong}."
            facts = search_kb(s["question"], s["country"], config.TOP_K)
        except Exception as exc:
            print(f"skip {s['question_id']}: {exc}")
            continue
        kb_has_gold = any(is_correct_saq(f.get("fact", ""), gold) for f in facts)

        placements = {
            "correct_in_english": dict(english=correct_en, local=gen["wrong_local"], local_en=wrong_en),
            "correct_in_local": dict(english=wrong_en, local=gen["correct_local"], local_en=correct_en),
        }
        for name, ans in placements.items():
            sim, flagged = check_disagreement(ans["local"], ans["english"])
            result = {"question_id": s["question_id"], "country": s["country"], "placement": name,
                      "similarity": sim, "flagged": int(flagged), "kb_has_gold": int(kb_has_gold)}
            for label, use_kb in (("nokb", False), ("kb", True)):
                prompt = judge.build_judge_prompt(
                    s["question"], ans["local"], ans["english"], facts if use_kb else [],
                    use_facts=use_kb, local_ans_en=ans["local_en"])
                try:
                    out = judge.run_judge(prompt)
                except Exception as exc:
                    print(f"judge failed {s['question_id']}: {exc}")
                    out = ""
                result[f"correct_{label}"] = int(judged_correct(out, gold, wrong))
            rows.append(result)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No conflict cases were generated.")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"conflict_{run_id}.csv"), index=False)

    pct = lambda col: f"{df[col].mean() * 100:.1f}%"
    lines = [f"CONFLICT TEST {run_id}: {len(df)} cases from {df['question_id'].nunique()} questions",
             f"Disagreement detection rate (similarity < {config.DISAGREEMENT_THRESHOLD}): {pct('flagged')}",
             f"Judge accuracy WITHOUT KB facts: {pct('correct_nokb')}",
             f"Judge accuracy WITH KB facts:    {pct('correct_kb')}",
             f"KB retrieved the correct answer: {pct('kb_has_gold')}",
             "", "By placement (accuracy %):",
             (df.groupby("placement")[["flagged", "correct_nokb", "correct_kb"]].mean() * 100).round(1).to_string(),
             "", "By country (accuracy %):",
             (df.groupby("country")[["flagged", "correct_nokb", "correct_kb", "kb_has_gold"]].mean() * 100).round(1).to_string(),
             "", "When the KB HAS the answer vs NOT (judge accuracy % with KB):",
             (df.groupby("kb_has_gold")[["correct_nokb", "correct_kb"]].mean() * 100).round(1).to_string()]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, f"conflict_{run_id}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL of short-answer questions with ground truth")
    p.add_argument("--run-id", default="conflict")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(data_loader.load_jsonl(a.input), a.run_id, a.out)
