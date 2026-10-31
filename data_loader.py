"""Loads evaluation questions: from a JSONL file, or from BLEnD on Hugging Face.

BLEnD layout (checked against the dataset card):
  * short answer : config "annotations", ONE SPLIT PER COUNTRY CODE (KR, ET, CN, ID, AZ).
                   Row = {ID, question (local), en_question, annotations: [{answers, en_answers, count}], idks}
  * multiple choice: config "multiple-choice-questions", split "test".
                   Row = {MCQID, ID, country, prompt, choices (JSON {"A": ..}), choice_countries, answer_idx}

Run `python adjudicator.py --show-samples` (and `--answer-type saq --show-samples`) to eyeball the result.

JSONL format (one object per line), for --input:
  {"question_id": "ko-1", "country": "South Korea", "answer_type": "saq",
   "question": "...", "local_question": "... (optional)", "ground_truth": ["answer", "alt"]}
For MCQ put the full text (stem + lettered options) in "question" and the letter in "ground_truth".
"""
import json
import re

import config

HF_NAMES = ["nayeon212/BLEnD", "uilab/BLEnD"]     # the dataset moved; try both


def _pick(row, *names, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


# ------------------------------------------------------------------ JSONL
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            country = config.normalize_country(row["country"])
            answer_type = row.get("answer_type", "saq")
            samples.append({
                "question_id": str(row.get("question_id", f"{country}-{i}")),
                "country": country,
                "language": row.get("language") or config.COUNTRY_LANGUAGE.get(country, "English"),
                "question": row["question"],
                "local_question": row.get("local_question"),
                "ground_truth": row["ground_truth"],
                "answer_type": answer_type,
                "answer_format": "mcq" if answer_type == "mcq" else "free",
            })
    return samples


# ------------------------------------------------------------------ Hugging Face
def _load_hf(config_name, split):
    from datasets import load_dataset
    last = None
    for name in HF_NAMES:
        try:
            return load_dataset(name, config_name, split=split)
        except Exception as exc:          # wrong repo name / split / network
            last = exc
    raise RuntimeError(f"Could not load BLEnD config={config_name!r} split={split!r}: {last}")


# ---- short answer (from the annotations config)
def _normalize_groups(groups):
    """`annotations` is a list of dicts, or (older `datasets`) a dict of lists. Return list of dicts."""
    if isinstance(groups, dict):
        keys = list(groups.keys())
        n = len(groups[keys[0]]) if keys else 0
        return [{k: groups[k][i] for k in keys} for i in range(n)]
    return list(groups or [])


def _gold_answers(groups):
    """English annotator answers. Keep answer groups with >= 2 votes; if none, keep the top group."""
    scored = []
    for g in _normalize_groups(groups):
        answers = g.get("en_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a).strip() for a in answers if str(a).strip()]
        if answers:
            scored.append((int(g.get("count") or 0), answers))
    if not scored:
        return []
    chosen = [a for count, a in scored if count >= 2] or [max(scored, key=lambda x: x[0])[1]]
    gold, seen = [], set()
    for answers in chosen:
        for a in answers:
            if a.lower() not in seen:
                seen.add(a.lower())
                gold.append(a)
    return gold


def _saq_samples():
    samples = []
    for country, code in config.COUNTRY_CODE.items():
        for i, row in enumerate(_load_hf("annotations", code)):
            english_q = _pick(row, "en_question")
            local_q = _pick(row, "question")
            gold = _gold_answers(row.get("annotations"))
            if not (english_q and gold):
                continue
            samples.append({
                "question_id": f"{code}-{_pick(row, 'ID', default=i)}",   # IDs repeat across countries
                "country": country, "language": config.COUNTRY_LANGUAGE[country],
                "question": english_q, "local_question": local_q,
                "ground_truth": gold, "answer_type": "saq", "answer_format": "free",
            })
    return samples


# ---- multiple choice
def _parse_choices(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    if isinstance(raw, dict):
        return {str(k).strip().upper(): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {chr(65 + i): str(v) for i, v in enumerate(raw)}
    return {}


def _stem(prompt):
    """BLEnD prompts are 'question? Without any explanation, choose only one ... \\nA. x\\nB. y'."""
    text = re.sub(r"^\s*question\s*:\s*", "", str(prompt), flags=re.I)
    text = re.split(r"\s*Without any explanation", text, maxsplit=1)[0]
    text = re.split(r"\n\s*\(?A[\.\):]", text, maxsplit=1)[0]
    return text.strip()


def _mcq_from_row(row, i):
    country = config.normalize_country(_pick(row, "country", "Country", default=""))
    if country not in config.COUNTRIES:
        return None
    choices = _parse_choices(_pick(row, "choices", "options"))
    stem = _stem(_pick(row, "prompt", "question", default=""))
    gold = _pick(row, "answer_idx", "answer", "label")
    if not (stem and choices and gold is not None):
        return None
    gold = chr(65 + gold) if isinstance(gold, int) else str(gold).strip().upper()[:1]
    if gold not in choices:
        return None
    text = stem + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    return {
        "question_id": f"{config.COUNTRY_CODE[country]}-{_pick(row, 'MCQID', 'ID', default=i)}",
        "base_id": str(_pick(row, "ID", default=i)),
        "country": country, "language": config.COUNTRY_LANGUAGE[country],
        "question": text, "local_question": None, "ground_truth": gold,
        "answer_type": "mcq", "answer_format": "mcq",
    }


def _mcq_samples():
    rows = _load_hf("multiple-choice-questions", "test")
    samples, seen = [], set()
    for i, row in enumerate(rows):
        s = _mcq_from_row(row, i)
        if not s:
            continue
        key = (s["country"], s["base_id"])      # BLEnD repeats a question with different distractors
        if key in seen:
            continue
        seen.add(key)
        samples.append(s)
    if not samples:
        countries = sorted({str(c) for c in rows["country"]})[:30] if "country" in rows.column_names else "?"
        raise RuntimeError(f"No MCQ rows matched our countries. Columns: {rows.column_names}. "
                           f"Country values seen: {countries}")
    return samples


# ------------------------------------------------------------------ public API
def _take(samples, per_country, skip):
    counts, kept = {}, []
    for s in samples:
        n = counts.get(s["country"], 0)
        counts[s["country"]] = n + 1
        if skip <= n < skip + per_country:
            kept.append(s)
    return kept


def load_samples(answer_type="mcq", per_country=20, skip=0, input_path=None):
    """Return `per_country` questions for each of the five countries (after skipping `skip`)."""
    if input_path:
        return _take(load_jsonl(input_path), per_country, skip)
    samples = _mcq_samples() if answer_type == "mcq" else _saq_samples()
    if not samples:
        raise RuntimeError("No usable samples were loaded from BLEnD. Pass --input your_file.jsonl")
    return _take(samples, per_country, skip)
