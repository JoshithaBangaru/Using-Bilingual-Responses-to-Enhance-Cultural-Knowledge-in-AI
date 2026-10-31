"""Dual-language generation: two INDEPENDENT calls (English and local language)."""
import json
import re

import config
import llm

_FORMAT_FREE = ("Give a short, direct answer that names the specific item or practice "
                "asked about, followed by one sentence of explanation.")
_FORMAT_MCQ = ("The question lists lettered options. Begin your answer with 'Answer: X' "
               "where X is the letter of the correct option, then give one sentence of "
               "justification.")


def get_country_language(country: str) -> str:
    return config.COUNTRY_LANGUAGE.get(country, "English")


def _parse_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def generate_bilingual_answers(question: str, country: str = "", language: str = "English",
                               local_question: str = None, answer_format: str = "free") -> dict:
    """Return local_answer (local language), local_answer_en (its English rendering,
    used for grading / the judge), english_answer and local_language.

    answer_format: "free" for short-answer questions, "mcq" for multiple choice.
    """
    local_lang = language if (language and language != "English") else get_country_language(country)
    where = f"in {country}" if country else "in the relevant country"
    fmt = _FORMAT_MCQ if answer_format == "mcq" else _FORMAT_FREE

    english_prompt = (
        f"You are an expert on everyday life and culture {where}.\n"
        f"Answer the question below in English. {fmt}\n\nQuestion:\n{question}"
    )
    english_answer = llm.generate(english_prompt)

    if local_lang == "English":
        return {"local_answer": english_answer, "local_answer_en": english_answer,
                "english_answer": english_answer, "local_language": local_lang}

    # MCQ options are written in English, so only free-form questions use the local text.
    local_q = local_question if (local_question and answer_format != "mcq") else question
    local_prompt = (
        f"You are an expert on everyday life and culture {where}.\n"
        f"Answer the question below. {fmt}\n"
        f"Write your answer in {local_lang}, and also give the same answer as a short "
        f"English translation.\n"
        f'Return ONLY a JSON object with the keys "answer" (in {local_lang}) and '
        f'"answer_en" (English translation).\n\nQuestion:\n{local_q}'
    )
    raw = llm.generate(local_prompt, json_mode=True)
    data = _parse_json(raw) or {}
    local_answer = str(data.get("answer") or raw).strip()
    local_answer_en = str(data.get("answer_en") or local_answer).strip()
    return {"local_answer": local_answer, "local_answer_en": local_answer_en,
            "english_answer": english_answer, "local_language": local_lang}
