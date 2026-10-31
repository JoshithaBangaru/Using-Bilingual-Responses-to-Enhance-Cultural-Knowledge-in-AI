"""Fill the evaluation table from the command line (same code as the web app's Batch tab).

    python batch_table.py --input eval_data/curated_questions.jsonl --run-id table1
    python batch_table.py --input my_questions.csv --country China --run-id table2
    python batch_table.py --blend saq --per-country 10 --skip 20 --run-id table3
    python batch_table.py --input q.txt --country "South Korea" --run-id table4     # lines: question | reference | category

Writes results/<run-id>_table.csv (and .xlsx if openpyxl is installed) and prints the summary.
"""
import argparse
import csv
import os

import config
import database
import table_utils
from records import build_record, rows_for_run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", help="CSV / JSONL / TXT file of questions")
    p.add_argument("--blend", choices=["mcq", "saq"], help="take questions from BLEnD instead")
    p.add_argument("--per-country", type=int, default=10)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--country", default="China", help="default country for files without a country column")
    p.add_argument("--mode", default="dual_kb", choices=config.MODES)
    p.add_argument("--run-id", default="table")
    p.add_argument("--no-evidence-for-all", action="store_true",
                   help="only show evidence when the judge actually used it")
    p.add_argument("--no-llm-assess", action="store_true",
                   help="leave Final Correct as 'Not graded' when no reference answer is given")
    p.add_argument("--no-cache", action="store_true", help="disable the LLM cache (true latency)")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    if a.no_cache:
        config.USE_CACHE = False
    if a.blend:
        from data_loader import load_samples
        items = [table_utils._item(s["question"], s["country"], s["ground_truth"], None,
                                   s["question_id"], s["answer_type"])
                 for s in load_samples(a.blend, a.per_country, a.skip)]
    elif a.input:
        with open(a.input, "rb") as f:
            items = table_utils.parse_upload(os.path.basename(a.input), f.read(), a.country)
    else:
        raise SystemExit("Give --input FILE or --blend mcq|saq")

    database.init_db()
    done = {r["question_id"] for r in database.fetch_runs(a.run_id) if not r.get("error")}
    print(f"{len(items)} questions, mode={a.mode}, run-id={a.run_id} ({len(done)} already done)")
    for i, item in enumerate(items, 1):
        qid = str(item["question_id"] or f"q{i}")
        if qid in done:
            continue
        try:
            out = build_record(item["question"], item["country"], mode=a.mode,
                               answer_type=item["answer_type"], reference=item["reference"],
                               category=item["category"], question_id=qid, run_id=a.run_id,
                               show_evidence=not a.no_evidence_for_all,
                               assess_without_reference=not a.no_llm_assess)
            row = out["row"]
            print(f"[{i}/{len(items)}] {row['Country/Region']}: {row['Final Correct']} "
                  f"({row['Correctness Basis']}), {row['Latency (ms)']} ms")
        except Exception as exc:
            print(f"[{i}/{len(items)}] FAILED {qid}: {exc}  (rerun the same command to retry)")

    rows = rows_for_run(a.run_id)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.run_id}_table.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=table_utils.TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=table_utils.TABLE_COLUMNS).to_excel(
            os.path.join(a.out, f"{a.run_id}_table.xlsx"), index=False)
    except Exception:
        pass

    s = table_utils.summarize(rows)
    fmt = lambda v, suffix="": "n/a" if v is None else f"{v:.1f}{suffix}"
    print(f"\nSaved {len(rows)} rows to {path}")
    print(f"Accuracy (reference-graded, n={s['n_reference']}): {fmt(s['accuracy_reference'], '%')}")
    print(f"Accuracy (LLM-assessed, n={s['n_llm_assessed']}, less reliable): {fmt(s['accuracy_llm_assessed'], '%')}")
    print(f"Disagreement rate: {fmt(s['disagreement_rate'], '%')} | KB grounded: {fmt(s['kb_grounded_rate'], '%')}")
    print(f"Latency median / mean: {fmt(s['latency_median_ms'], ' ms')} / {fmt(s['latency_mean_ms'], ' ms')}")
    print("Accuracy by country:", table_utils.accuracy_by(rows, "Country/Region"))


if __name__ == "__main__":
    main()
