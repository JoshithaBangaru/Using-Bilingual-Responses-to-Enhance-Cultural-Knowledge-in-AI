"""Pre-flight check. Run this first:  python check_setup.py"""
import json
import os

import config

failures = 0


def check(label, fn):
    global failures
    try:
        print(f"[ OK ] {label}: {fn()}")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] {label}: {exc}")


def api_key():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return "set"


def model_listed():
    import llm
    names = llm.list_models()
    wanted = config.GEMINI_MODEL
    found = any(n == wanted or n.endswith("/" + wanted) for n in names)
    if not found:
        sample = ", ".join(n for n in names if "gemini" in n)[:400]
        raise RuntimeError(f"'{wanted}' not available to this key. Gemini models: {sample}")
    return f"{wanted} available ({len(names)} models listed)"


def model_call():
    import llm
    return llm.generate("Reply with the single word OK.", use_cache=False)[:40]


def kb_files():
    for path in (config.DATA_PATH, config.INDEX_PATH, config.METADATA_PATH):
        if not os.path.exists(path):
            raise RuntimeError(f"missing {os.path.basename(path)} (run the scraper and kb_retriever.py)")
    return "cultural_data.json, faiss_index.bin, metadata.json present"


def kb_consistency():
    import faiss
    index = faiss.read_index(config.INDEX_PATH)
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    if index.ntotal != len(meta):
        raise RuntimeError(f"index={index.ntotal} vectors but metadata={len(meta)} entries")
    counts = {c: sum(1 for m in meta if m.get("country") == c) for c in config.COUNTRIES}
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise RuntimeError(f"no facts for {missing}")
    thin = [c for c, n in counts.items() if n < 50]
    note = f" (WARNING: fewer than 50 facts for {thin})" if thin else ""
    return f"{index.ntotal} facts {counts}{note}"


def database():
    import database as db
    db.init_db()
    return config.DB_PATH


if __name__ == "__main__":
    print(f"Model: {config.GEMINI_MODEL} | threshold: {config.DISAGREEMENT_THRESHOLD}\n")
    check("GEMINI_API_KEY", api_key)
    check("Model available", model_listed)
    check("Test generation", model_call)
    check("KB files", kb_files)
    check("KB index/metadata", kb_consistency)
    check("SQLite", database)
    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed - fix these first.")
