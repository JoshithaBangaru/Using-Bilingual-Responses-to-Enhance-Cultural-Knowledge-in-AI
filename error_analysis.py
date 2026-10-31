"""Failure analysis for the report's error-analysis section.

    python error_analysis.py --run-id mcq_test --mode dual_kb --n 15

Every wrong answer gets a category:
  agreed_but_wrong     the two answers agreed, so the judge/KB never ran
  judge_broke_correct  a single answer was right but the judge's final answer was wrong
  kb_miss              both answers wrong and the retrieved facts do not contain the gold answer (short answer)
  judge_ignored_kb     both answers wrong, retrieved facts DID contain the gold answer, judge still wrong
  both_wrong_check_kb  multiple choice: both wrong; read the retrieved facts to decide KB miss vs judge error
Also reports KB answer-coverage and how often the correct final answer was grounded in retrieved facts.
"""
import argparse
import json
import os

import pandas as pd

import database
from textutils import is_correct_saq, to_answer_list


def _facts_text(raw):
    try:
        facts = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except ValueError:
        return ""
    return " ".join(str(f.get("fact", "")) for f in facts if isinstance(f, dict))


def categorize(row):
    if row["is_disagreed"] != 1:
        return "agreed_but_wrong"
    if row["english_correct"] == 1 or row["local_correct"] == 1:
        return "judge_broke_correct"
    if row["answer_type"] == "saq":
        has_gold = is_correct_saq(_facts_text(row["retrieved_facts"]), to_answer_list(row["ground_truth"]))
        return "judge_ignored_kb" if has_gold else "kb_miss"
    return "both_wrong_check_kb"


def run(run_id, mode, n, out_dir):
    df = pd.DataFrame(database.fetch_runs(run_id))
    if df.empty:
        raise SystemExit(f"No rows for run {run_id!r}")
    df["error"] = df["error"].fillna("")
    df = df[(df["error"] == "") & (df["mode"] == mode)]
    if df.empty:
        raise SystemExit(f"No successful rows for mode {mode!r}")

    wrong = df[df["is_correct"] == 0].copy()
    wrong["category"] = wrong.apply(categorize, axis=1)

    print(f"Run {run_id} / {mode}: {len(df)} questions, {len(wrong)} wrong "
          f"({len(wrong) / len(df) * 100:.1f}%)\n")
    print("FAILURE CATEGORIES")
    print(wrong["category"].value_counts().to_string(), "\n")
    print("FAILURES BY COUNTRY")
    print(pd.crosstab(wrong["country"], wrong["category"]).to_string(), "\n")

    judged = df[df["is_disagreed"] == 1]
    saq = judged[judged["answer_type"] == "saq"]
    if not saq.empty:
        covered = saq.apply(lambda r: is_correct_saq(_facts_text(r["retrieved_facts"]),
                                                     to_answer_list(r["ground_truth"])), axis=1)
        print(f"KB ANSWER-COVERAGE (short answer, judged cases): {covered.mean() * 100:.1f}% of retrievals "
              f"contain the gold answer")
        grounded = (covered & (saq["is_correct"] == 1)).sum()
        right = (saq["is_correct"] == 1).sum()
        if right:
            print(f"KB-GROUNDED correct answers: {grounded}/{right} ({grounded / right * 100:.1f}%)\n")

    os.makedirs(out_dir, exist_ok=True)
    cols = ["question_id", "country", "category", "question", "ground_truth", "english_answer",
            "local_answer", "similarity", "retrieved_facts", "final_answer"]
    wrong[cols].to_csv(os.path.join(out_dir, f"{run_id}_{mode}_failures.csv"), index=False)
    print(f"All failures saved to {out_dir}/{run_id}_{mode}_failures.csv\n")

    print(f"=== {min(n, len(wrong))} RANDOM FAILURES TO READ AND DISCUSS IN THE REPORT ===")
    for _, r in wrong.sample(min(n, len(wrong)), random_state=0).iterrows():
        print("-" * 72)
        print(f"[{r['country']}] {r['category']} (similarity={r['similarity']})")
        print("Q:", str(r["question"])[:300])
        print("GOLD:", r["ground_truth"])
        print("ENGLISH:", str(r["english_answer"])[:200])
        print("FINAL:", str(r["final_answer"])[:200])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", default="dual_kb")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.run_id, a.mode, a.n, a.out)
