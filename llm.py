"""Single entry point for every Gemini call: retries, back-off and an on-disk cache.

Errors are RAISED (LLMError) instead of being returned as answer text, so a failed
call can never be logged or graded as if it were a real answer.
"""
import hashlib
import sqlite3
import time

import config


class LLMError(RuntimeError):
    pass


_backend = None
_client = None
_legacy = None


def _init_backend():
    global _backend, _client, _legacy
    if _backend:
        return
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set (export it or put it in a .env file).")
    try:
        from google import genai            # current SDK: pip install google-genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
        _backend = "genai"
    except ImportError:
        try:
            import google.generativeai as legacy   # deprecated SDK, still supported here
        except ImportError as exc:
            raise LLMError("Install the Gemini SDK: pip install google-genai") from exc
        legacy.configure(api_key=config.GEMINI_API_KEY)
        _legacy = legacy
        _backend = "legacy"


# ------------------------------------------------------------------ cache
def _cache_conn():
    conn = sqlite3.connect(config.CACHE_DB_PATH, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache ("
        "key TEXT PRIMARY KEY, model TEXT, response TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def _key(prompt, json_mode):
    raw = f"{config.GEMINI_MODEL}|{int(json_mode)}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key):
    conn = _cache_conn()
    try:
        row = conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _cache_set(key, response):
    conn = _cache_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, model, response) VALUES (?, ?, ?)",
            (key, config.GEMINI_MODEL, response),
        )
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ calls
def _raw_call(prompt, json_mode):
    if _backend == "genai":
        cfg = {"response_mime_type": "application/json"} if json_mode else None
        response = _client.models.generate_content(
            model=config.GEMINI_MODEL, contents=prompt, config=cfg
        )
        return response.text
    model = _legacy.GenerativeModel(config.GEMINI_MODEL)
    kwargs = {"generation_config": {"response_mime_type": "application/json"}} if json_mode else {}
    return model.generate_content(prompt, **kwargs).text


_RETRYABLE = ("429", "quota", "rate", "503", "500", "overloaded",
              "unavailable", "resource_exhausted", "deadline", "timeout")


def _is_retryable(exc):
    text = str(exc).lower()
    return any(token in text for token in _RETRYABLE)


def generate(prompt, json_mode=False, use_cache=None):
    """Return the model's text for `prompt`; raise LLMError if it cannot be obtained."""
    use_cache = config.USE_CACHE if use_cache is None else use_cache
    key = _key(prompt, json_mode)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit

    _init_backend()
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            text = _raw_call(prompt, json_mode)
        except Exception as exc:  # SDK-specific exception types differ between SDKs
            if _is_retryable(exc) and attempt < config.LLM_MAX_RETRIES - 1:
                time.sleep(min(15 * (2 ** attempt), 120))
                continue
            raise LLMError(f"Gemini call failed ({config.GEMINI_MODEL}): {exc}") from exc
        text = (text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response (possibly a safety block).")
        if use_cache:
            _cache_set(key, text)
        return text
    raise LLMError("Gemini call failed: retries exhausted.")


def list_models():
    _init_backend()
    if _backend == "genai":
        return [m.name for m in _client.models.list()]
    return [m.name for m in _legacy.list_models()]
