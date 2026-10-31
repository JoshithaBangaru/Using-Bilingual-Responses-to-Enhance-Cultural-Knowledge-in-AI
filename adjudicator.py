"""Benchmark runner: evaluates every pipeline variant on BLEnD questions.

    python adjudicator.py --answer-type mcq --samples-per-country 20 --run-id mcq_v1
    python adjudicator.py --input eval_data/sample_eval.jsonl --run-id smoke      # offline smoke test
    python adjudicator.py --show-samples                                           # inspect loaded data

Runs are resumable: re-running with the same --run-id only repeats missing / errored rows.
Calls pipeline.run_pipeline directly (no HTTP server needed).
"""
import argparse
import sys
from datetime import datetime

from tqdm import tqdm

import config
import database
from data_loader import load_samples
from llm import LLMError
from textutils import is_correct, to_answer_list


def parse_modes(arg):
    if arg == "all":
        return list(config.MODES)
    modes = [m.strip() for m in arg.split(",") if m.strip()]
    bad = [m for m in modes if m not in config.MODES]
    if bad:
        sys.exit(f"Unknown modes {bad}. Valid: {config.MODES}")
    return modes


def evaluate(samples, modes, run_id):
    from pipeline import run_pipeline          # heavy imports only when actually evaluating

    database.init_db()
    errors = 0
    for sample in tqdm(samples, desc="Evaluating"):
        gt = sample["ground_truth"]
        gt_store = gt if isinstance(gt, str) else to_answer_list(gt)
        for mode in modes:
            if database.already_logged(run_id, sample["question_id"], mode):
                continue
            database.delete_errors(run_id, sample["question_id"], mode)
            record = {
                "run_id": run_id, "mode": mode, "question_id": sample["question_id"],
                "country": sample["country"], "language": sample["language"],
                "answer_type": sample["answer_type"], "question": sample["question"],
                "ground_truth": gt_store,
            }
            try:
                res = run_pipeline(
                    question=sample["question"], country=sample["country"],
                    language=sample["language"], mode=mode,
                    answer_format=sample["answer_format"],
                    local_question=sample.get("local_question"))
            except (LLMError, Exception) as exc:     # keep going; failed rows are retried on re-run
                record["error"] = f"{type(exc).__name__}: {exc}"
                database.log_run(record)
                errors += 1
                continue

            kind = sample["answer_type"]
            record.update({
                "local_answer": res["local_answer"], "local_answer_en": res["local_answer_en"],
                "english_answer": res["english_answer"], "similarity": res["similarity_score"],
                "is_disagreed": res["is_disagreed"], "used_judge": res["used_judge"],
                "retrieved_facts": res["retrieved_facts"], "final_answer": res["final_answer"],
                "is_correct": is_correct(res["final_answer_en"], gt, kind),
                "english_correct": is_correct(res["english_answer"], gt, kind),
                "local_correct": is_correct(res["local_answer_en"], gt, kind),
                "latency_ms": res["latency_ms"],
            })
            database.log_run(record)
    if errors:
        print(f"\n{errors} rows failed (rate limits / API errors). Re-run the same command to retry them.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--answer-type", choices=["mcq", "saq"], default="mcq")
    p.add_argument("--samples-per-country", type=int, default=20)
    p.add_argument("--skip", type=int, default=0,
                   help="skip the first N questions per country (use a dev slice to tune the threshold, "
                        "then evaluate on a disjoint test slice)")
    p.add_argument("--modes", default="all", help="'all' or a comma-separated list")
    p.add_argument("--input", help="JSONL file instead of Hugging Face BLEnD")
    p.add_argument("--run-id", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    p.add_argument("--show-samples", action="store_true", help="print a few loaded samples and exit")
    p.add_argument("--no-analyze", action="store_true")
    args = p.parse_args()

    samples = load_samples(args.answer_type, args.samples_per_country, args.skip, args.input)
    per_country = {c: sum(1 for s in samples if s["country"] == c) for c in config.COUNTRIES}
    print(f"Loaded {len(samples)} questions: {per_country}")

    if args.show_samples:
        for s in samples[:5]:
            print("-" * 70)
            print(f"[{s['country']}] id={s['question_id']} type={s['answer_type']}")
            print(s["question"])
            print("GOLD:", s["ground_truth"])
        return

    modes = parse_modes(args.modes)
    print(f"Run id: {args.run_id} | modes: {modes} | model: {config.GEMINI_MODEL}")
    evaluate(samples, modes, args.run_id)

    if not args.no_analyze:
        import analyze
        analyze.run(args.run_id)


if __name__ == "__main__":
    main()
