"""Pure-Python helpers (no heavy imports) shared by the pipeline, evaluation and tests."""
import json
import re
import unicodedata


def extract_text(item):
    """Text of a KB entry regardless of key name (the KB stores it under 'fact')."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("fact", "text", "content", "snippet", "passage", "summary"):
            if item.get(key):
                return str(item[key])
        return " ".join(str(v) for v in item.values() if v)
    return str(item)


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_answer_list(ground_truth):
    """Accept a list, a JSON list string, 'a||b', or a single string."""
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple, set)):
        return [str(a) for a in ground_truth if str(a).strip()]
    s = str(ground_truth).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(a) for a in parsed if str(a).strip()]
        except ValueError:
            pass
    if "||" in s:
        return [p.strip() for p in s.split("||") if p.strip()]
    return [s]


_LETTER_PATTERNS = [
    re.compile(r"(?i:answer)\s*(?:is\s*)?[:\-]?\s*\(?([A-D])\)?(?![A-Za-z])"),
    re.compile(r"^\W*\(?([A-D])[\.\)]"),
]


def extract_letter(text):
    """Option letter (A-D) from 'Answer: B. ...' style output, else None."""
    if not text:
        return None
    cleaned = re.sub(r"[*`_]", "", str(text)).strip()
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return match.group(1)
    return None


def is_correct_mcq(prediction, gold_letter):
    gold = str(gold_letter or "").strip().upper()[:1]
    return bool(gold) and extract_letter(prediction) == gold


def is_correct_saq(prediction, answers):
    """True if any accepted answer appears in the prediction (normalised, whole-word)."""
    pred = normalize(prediction)
    if not pred:
        return False
    for answer in to_answer_list(answers):
        target = normalize(answer)
        if not target:
            continue
        if target.isascii():
            if re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", pred):
                return True
        elif target in pred:      # CJK / Ethiopic text has no word boundaries
            return True
    return False


def is_correct(prediction, ground_truth, answer_type="saq"):
    if answer_type == "mcq":
        return is_correct_mcq(prediction, ground_truth)
    return is_correct_saq(prediction, ground_truth)
