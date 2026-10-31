"""Turns logged evaluation rows into the tables and plots for the report.

    python analyze.py --run-id mcq_v1 --out results
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

import config
import database

JUDGE_MODES = ["dual_judge_nokb", "dual_kb", "dual_kb_always"]
DUAL_MODES = JUDGE_MODES


def load_df(run_id):
    rows = database.fetch_runs(run_id)
    if not rows:
        raise SystemExit(f"No rows found for run_id={run_id!r}. Known runs: {database.list_run_ids()}")
    df = pd.DataFrame(rows)
    df["error"] = df["error"].fillna("")
    return df


def _order(columns):
    return [m for m in config.MODES if m in columns]


def accuracy_tables(ok):
    overall = ok.groupby("mode")["is_correct"].agg(["count", "mean"])
    overall["accuracy_%"] = overall["mean"] * 100
    overall = overall.drop(columns="mean").reindex(_order(overall.index))
    by_country = ok.pivot_table(index="country", columns="mode", values="is_correct", aggfunc="mean") * 100
    by_country = by_country[_order(by_country.columns)]
    spread = (by_country.max() - by_country.min()).rename("country_spread_pts")
    gap = None
    if {"english_only", "local_only"} <= set(by_country.columns):
        gap = (by_country["english_only"] - by_country["local_only"]).rename("english_minus_local_pts")
    return overall, by_country, spread, gap


def adjudication_stats(ok):
    """On disagreement cases: how often does the judge fix things (rescue) or break them (harm)?"""
    rows = []
    for mode in JUDGE_MODES:
        d = ok[(ok["mode"] == mode) & (ok["is_disagreed"] == 1)]
        if d.empty:
            continue
        both_wrong = (d["english_correct"] == 0) & (d["local_correct"] == 0)
        any_right = (d["english_correct"] == 1) | (d["local_correct"] == 1)
        rows.append({
            "mode": mode, "n_disagreed": len(d),
            "english_acc_%": d["english_correct"].mean() * 100,
            "local_acc_%": d["local_correct"].mean() * 100,
            "judge_final_acc_%": d["is_correct"].mean() * 100,
            "adjudication_success_%": ((d["is_correct"] == 1) & both_wrong).mean() * 100,
            "harm_%": ((d["is_correct"] == 0) & any_right).mean() * 100,
        })
    return pd.DataFrame(rows).set_index("mode") if rows else None


def bootstrap_ci(values, n_boot=2000, seed=0):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5]) * 100
    return float(low), float(high)


def ci_table(ok):
    rows = {}
    for mode in _order(ok["mode"].unique()):
        vals = ok[ok["mode"] == mode]["is_correct"].astype(float)
        low, high = bootstrap_ci(vals)
        rows[mode] = {"n": len(vals), "accuracy_%": vals.mean() * 100,
                      "ci95_low_%": low, "ci95_high_%": high}
    return pd.DataFrame(rows).T


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired 0/1 outcomes. Returns (a_only, b_only, p)."""
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return a_only, b_only, min(1.0, p)


def significance_table(ok):
    wide = ok.pivot_table(index="question_id", columns="mode", values="is_correct", aggfunc="first")
    rows = []
    for other in ("english_only", "local_only", "dual_judge_nokb", "dual_kb_always"):
        if "dual_kb" in wide.columns and other in wide.columns:
            pair = wide[["dual_kb", other]].dropna()
            if pair.empty:
                continue
            kb_only, other_only, p = mcnemar_exact(pair["dual_kb"], pair[other])
            rows.append({"comparison": f"dual_kb vs {other}", "n_pairs": len(pair),
                         "dual_kb_only_right": kb_only, "other_only_right": other_only,
                         "p_value": p})
    return pd.DataFrame(rows).set_index("comparison") if rows else None


def resource_level_table(ok):
    ok = ok.copy()
    ok["resource_level"] = ok["country"].map(config.RESOURCE_LEVEL)
    table = ok.pivot_table(index="resource_level", columns="mode", values="is_correct",
                           aggfunc="mean") * 100
    return table[_order(table.columns)]


def robustness_table(ok):
    """Accuracy per question variant (original / -para / -xling); needs augmented rows in the same run."""
    ok = ok.copy()
    ok["variant"] = ok["question_id"].astype(str).str.extract(r"-(para|xling)$")[0].fillna("original")
    if ok["variant"].nunique() < 2:
        return None
    table = ok.pivot_table(index="variant", columns="mode", values="is_correct", aggfunc="mean") * 100
    return table[_order(table.columns)]


def threshold_sweep(ok):
    dual = ok[ok["mode"].isin(DUAL_MODES) & ok["similarity"].notna()].drop_duplicates("question_id")
    if dual.empty:
        return None, dual
    english_wrong = dual["english_correct"] == 0
    rows = []
    for t in np.arange(0.5, 0.9751, 0.025):
        flagged = dual["similarity"] < t
        rows.append({
            "threshold": round(float(t), 3),
            "disagreement_rate_%": flagged.mean() * 100,
            "precision_%": english_wrong[flagged].mean() * 100 if flagged.any() else np.nan,
            "recall_%": flagged[english_wrong].mean() * 100 if english_wrong.any() else np.nan,
        })
    return pd.DataFrame(rows), dual


def make_plots(out_dir, by_country, dual, sweep, run_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    ax = by_country.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy by country and pipeline variant ({run_id})")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    plt.tight_layout()
    paths.append(os.path.join(out_dir, "accuracy_by_country.png"))
    plt.savefig(paths[-1], dpi=150)
    plt.close()

    if dual is not None and not dual.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        right = dual[dual["english_correct"] == 1]["similarity"]
        wrong = dual[dual["english_correct"] == 0]["similarity"]
        ax.hist([right, wrong], bins=20, label=["English answer correct", "English answer wrong"])
        ax.axvline(config.DISAGREEMENT_THRESHOLD, color="k", linestyle="--", label="threshold")
        ax.set_xlabel("Cosine similarity (local vs English answer)")
        ax.set_ylabel("Questions")
        ax.legend()
        plt.tight_layout()
        paths.append(os.path.join(out_dir, "similarity_hist.png"))
        plt.savefig(paths[-1], dpi=150)
        plt.close()

    if sweep is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for col in ["disagreement_rate_%", "precision_%", "recall_%"]:
            ax.plot(sweep["threshold"], sweep[col], marker="o", label=col)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("%")
        ax.legend()
        plt.tight_layout()
        paths.append(os.path.join(out_dir, "threshold_sweep.png"))
        plt.savefig(paths[-1], dpi=150)
        plt.close()
    return paths


def _fmt(df):
    return df.to_string(float_format=lambda x: f"{x:.1f}")


def run(run_id=None, out_dir="results"):
    run_id = run_id or database.list_run_ids()[0]
    os.makedirs(out_dir, exist_ok=True)
    df = load_df(run_id)
    ok = df[df["error"] == ""].copy()
    lines = [f"RUN {run_id} | rows={len(df)} | errors={int((df['error'] != '').sum())} | model={config.GEMINI_MODEL}"]
    if ok.empty:
        print("\n".join(lines + ["No successful rows to analyse."]))
        return

    overall, by_country, spread, gap = accuracy_tables(ok)
    lines += ["", "OVERALL ACCURACY BY VARIANT", _fmt(overall),
              "", "ACCURACY (%) BY COUNTRY", _fmt(by_country),
              "", "COUNTRY SPREAD (max - min accuracy across countries; smaller = smaller gap)",
              _fmt(spread.to_frame())]
    if gap is not None:
        lines += ["", "LANGUAGE GAP PER COUNTRY (English-only minus local-only, points)", _fmt(gap.to_frame())]

    cis = ci_table(ok)
    lines += ["", "ACCURACY WITH 95% BOOTSTRAP CONFIDENCE INTERVALS", _fmt(cis)]
    sig = significance_table(ok)
    if sig is not None:
        lines += ["", "PAIRED SIGNIFICANCE (exact McNemar, dual_kb vs others; p<0.05 = real difference)", _fmt(sig)]
    level = resource_level_table(ok)
    lines += ["", "ACCURACY (%) BY LANGUAGE RESOURCE LEVEL", _fmt(level)]
    rob = robustness_table(ok)
    if rob is not None:
        lines += ["", "ROBUSTNESS: ACCURACY (%) BY QUESTION VARIANT", _fmt(rob)]
        rob.to_csv(os.path.join(out_dir, f"{run_id}_robustness.csv"))
    cis.to_csv(os.path.join(out_dir, f"{run_id}_ci.csv"))
    level.to_csv(os.path.join(out_dir, f"{run_id}_resource_level.csv"))
    if sig is not None:
        sig.to_csv(os.path.join(out_dir, f"{run_id}_significance.csv"))

    adj = adjudication_stats(ok)
    if adj is not None:
        lines += ["", "ADJUDICATION ON DISAGREEMENT CASES", _fmt(adj)]

    main = ok[ok["mode"] == "dual_kb"]
    if not main.empty:
        lines += ["", f"Disagreement trigger rate (dual_kb): {(main['is_disagreed'] == 1).mean() * 100:.1f}%"]
    lat = ok.groupby("mode")["latency_ms"].agg(["mean", "median"]).reindex(_order(ok["mode"].unique()))
    lines += ["", "LATENCY (ms; use LLM_CACHE=0 for honest numbers)", _fmt(lat)]

    sweep, dual = threshold_sweep(ok)
    if sweep is not None:
        sweep.to_csv(os.path.join(out_dir, f"{run_id}_threshold_sweep.csv"), index=False)
    overall.to_csv(os.path.join(out_dir, f"{run_id}_overall.csv"))
    by_country.to_csv(os.path.join(out_dir, f"{run_id}_by_country.csv"))
    if adj is not None:
        adj.to_csv(os.path.join(out_dir, f"{run_id}_adjudication.csv"))
    plots = make_plots(out_dir, by_country, dual, sweep, run_id)

    text = "\n".join(lines)
    with open(os.path.join(out_dir, f"{run_id}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"\nSaved tables and {len(plots)} plots to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run(a.run_id, a.out)
