"""LLM-as-Judge. Receives both answers and (optionally) retrieved KB facts."""
import re

import llm
from textutils import extract_text


def _fact_line(item):
    topic = item.get("topic") if isinstance(item, dict) else None
    text = extract_text(item).strip()
    return f"- ({topic}) {text}" if topic else f"- {text}"


def build_judge_prompt(question: str, local_ans: str, english_ans: str, facts: list,
                       answer_format: str = "free", use_facts: bool = True,
                       local_ans_en: str = None) -> str:
    local_block = local_ans
    if local_ans_en and local_ans_en.strip() != local_ans.strip():
        local_block += f"\n(English rendering: {local_ans_en})"

    if not use_facts:
        facts_block = ""
    elif facts:
        lines = "\n".join(_fact_line(item) for item in facts)
        facts_block = (
            "\nReference facts retrieved from a cultural knowledge base (Wikipedia). "
            "They may be incomplete or only loosely related; use them only when they "
            f"bear on the question:\n{lines}\n"
        )
    else:
        facts_block = "\nNo relevant reference facts were retrieved.\n"

    if answer_format == "mcq":
        task = ("Decide which option is correct. Begin your reply with 'Answer: X' where X "
                "is the option letter, then give one sentence of justification.")
    else:
        task = ("Decide which answer is correct, or whether neither is, and give the final "
                "answer in English: a short, direct answer that names the specific item "
                "asked for, then one or two sentences of cultural context.")

    return (
        "You are a cultural knowledge adjudicator.\n"
        "A question about everyday culture was answered twice: once in the local language "
        "and once in English. The two answers differ.\n\n"
        f"Question:\n{question}\n\n"
        f"Local-language answer:\n{local_block}\n\n"
        f"English answer:\n{english_ans}\n"
        f"{facts_block}\n"
        f"{task} Do not invent facts you are not confident about.\n"
        "Finish with one last line exactly in the form 'Verdict: ENGLISH', 'Verdict: LOCAL' or "
        "'Verdict: NEITHER', saying which answer your final answer agrees with "
        "(NEITHER if you wrote a different answer)."
    )


_VERDICT = re.compile(r"\s*\**verdict\**\s*:\s*\**\s*(ENGLISH|LOCAL|NEITHER)\b[^\n]*\s*$", re.I)
_DECISIONS = {"ENGLISH": "Chose English answer", "LOCAL": "Chose local-language answer",
              "NEITHER": "Neither (wrote own answer)"}


def run_judge_full(prompt: str):
    """Return (final answer text without the verdict line, human-readable judge decision)."""
    text = llm.generate(prompt)
    match = _VERDICT.search(text)
    if not match:
        return text, "Unclear"
    answer = text[:match.start()].strip() or text
    return answer, _DECISIONS[match.group(1).upper()]


def run_judge(prompt: str) -> str:
    return run_judge_full(prompt)[0]
