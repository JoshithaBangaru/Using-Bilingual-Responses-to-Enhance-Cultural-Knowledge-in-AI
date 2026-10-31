"""Retrieval-relevance evaluation (hand-labelled).

    python label_retrieval.py export --input eval_data/sample_eval.jsonl --n 50 --out retrieval_labels.csv
    # open the CSV, put 1 (relevant) or 0 (not relevant) in the 'relevant' column
    python label_retrieval.py score --file retrieval_labels.csv
"""
import argparse
import csv

import config


def export(args):
    from data_loader import load_samples
    from kb_retriever import search_kb

    samples = load_samples(args.answer_type, args.n, 0, args.input)[: args.n]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "country", "question", "rank", "result_country",
                         "score", "topic", "fact", "source", "relevant"])
        for s in samples:
            for rank, hit in enumerate(search_kb(s["question"], s["country"], config.TOP_K), 1):
                writer.writerow([s["question_id"], s["country"], s["question"].replace("\n", " "),
                                 rank, hit.get("country"), round(hit["score"], 4),
                                 hit.get("topic"), hit.get("fact"), hit.get("source"), ""])
    print(f"Wrote {args.out}. Fill the 'relevant' column with 1/0, then run: score --file {args.out}")


def score(args):
    with open(args.file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if r["relevant"].strip() in ("0", "1")]
    if not labelled:
        raise SystemExit("No labelled rows found.")
    precision = sum(r["relevant"].strip() == "1" for r in labelled) / len(labelled)
    country_ok = sum(r["country"].lower() == r["result_country"].lower() for r in rows) / len(rows)
    queries_with_hit = {}
    for r in labelled:
        queries_with_hit.setdefault(r["question_id"], False)
        if r["relevant"].strip() == "1":
            queries_with_hit[r["question_id"]] = True
    hit_rate = sum(queries_with_hit.values()) / len(queries_with_hit)
    print(f"Labelled results: {len(labelled)}")
    print(f"Relevance precision@{config.TOP_K}: {precision * 100:.1f}%")
    print(f"Questions with >=1 relevant fact (hit rate): {hit_rate * 100:.1f}%")
    print(f"Country precision@{config.TOP_K}: {country_ok * 100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--input")
    e.add_argument("--answer-type", choices=["mcq", "saq"], default="saq")
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--out", default="retrieval_labels.csv")
    e.set_defaults(func=export)
    s = sub.add_parser("score")
    s.add_argument("--file", required=True)
    s.set_defaults(func=score)
    a = ap.parse_args()
    a.func(a)
