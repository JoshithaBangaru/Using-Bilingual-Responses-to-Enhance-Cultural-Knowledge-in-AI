"""Question augmentation (proposal: question paraphrasing + cross-lingual paraphrasing).

    python augment.py --input eval_data/curated_questions.jsonl --out eval_data/augmented.jsonl
    python adjudicator.py --input eval_data/curated_questions.jsonl --run-id aug --samples-per-country 50
    python adjudicator.py --input eval_data/augmented.jsonl         --run-id aug --samples-per-country 100
    python analyze.py --run-id aug      # the ROBUSTNESS table compares original / -para / -xling

For every question it writes up to two new rows (same ground truth):
  <id>-para : English paraphrase (same meaning, different wording)
  <id>-xling: the original English question + a local-language version of it (used by dual_gen
              for the local-language answer; short-answer questions only)
"""
import argparse
import json

import config
import data_loader
import llm

PARA_PROMPT = ("Rewrite the question below so that it keeps exactly the same meaning and the same "
               "single correct answer, but uses clearly different wording. Do not reveal the answer. "
               "Return ONLY the rewritten question.\n\nQuestion:\n{q}")
XLING_PROMPT = ("Translate the question below into {lang}. Keep the meaning exactly. "
                "Return ONLY the translation.\n\nQuestion:\n{q}")


def split_mcq(text):
    """Return (stem, options_text) for 'stem\\nA. ...\\nB. ...' formatted questions."""
    parts = text.split("\n", 1)
    return (parts[0], "\n" + parts[1]) if len(parts) == 2 else (text, "")


def augment(samples, kinds):
    out = []
    for s in samples:
        base = {k: s[k] for k in ("country", "language", "ground_truth", "answer_type")}
        if "paraphrase" in kinds:
            if s["answer_type"] == "mcq":
                stem, options = split_mcq(s["question"])
                new_q = llm.generate(PARA_PROMPT.format(q=stem)).strip() + options
            else:
                new_q = llm.generate(PARA_PROMPT.format(q=s["question"])).strip()
            out.append({**base, "question_id": f"{s['question_id']}-para", "question": new_q})
        if "crosslingual" in kinds and s["answer_type"] == "saq":
            local_q = llm.generate(XLING_PROMPT.format(lang=s["language"], q=s["question"])).strip()
            out.append({**base, "question_id": f"{s['question_id']}-xling",
                        "question": s["question"], "local_question": local_q})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="eval_data/augmented.jsonl")
    p.add_argument("--kinds", default="paraphrase,crosslingual")
    a = p.parse_args()
    samples = data_loader.load_jsonl(a.input)
    rows = augment(samples, set(k.strip() for k in a.kinds.split(",")))
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} augmented questions to {a.out}")
