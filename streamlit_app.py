"""Web app: ask questions, fill the evaluation table in bulk, and review saved runs.

Start the API first (uvicorn app:app --port 8000), then:  streamlit run streamlit_app.py
"""
import io
import json
import time

import pandas as pd
import requests
import streamlit as st

import config
import table_utils as tu

st.set_page_config(page_title="Cultural QA Evaluation", layout="wide")
st.title("Cultural QA: dual-language ensemble with KB-assisted judge")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API address", "http://127.0.0.1:8000").rstrip("/")
    mode = st.selectbox("Pipeline variant", config.MODES, index=config.MODES.index("dual_kb"))
    show_evidence = st.checkbox("Show retrieved evidence for every question", True,
                                help="Otherwise evidence appears only when the judge used it.")
    assess = st.checkbox("LLM-assess correctness when no reference answer is given", True,
                         help="Less reliable: the model grades its own knowledge. These rows are labelled "
                              "'LLM-assessed' and counted separately from reference-graded accuracy.")
    no_cache = st.checkbox("Disable cache (true latency)", False,
                           help="Slower and uses more API calls, but gives honest latency numbers.")
    st.caption(f"Disagreement threshold: {config.DISAGREEMENT_THRESHOLD}")


def call_record(item, run_id):
    payload = {
        "question": item["question"], "country": item["country"], "mode": mode,
        "answer_type": item.get("answer_type"),
        "reference": "||".join(item["reference"]) if item.get("reference") else None,
        "category": item.get("category"), "question_id": item.get("question_id"),
        "run_id": run_id, "show_evidence": show_evidence,
        "assess_without_reference": assess, "no_cache": no_cache,
    }
    response = requests.post(f"{api_url}/record", json=payload, timeout=900)
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:300]}")
    return response.json()


def show_summary(rows, timings=None):
    s = tu.summarize(rows)
    fmt = lambda v, suffix="": "n/a" if v is None else f"{v:.1f}{suffix}"
    c = st.columns(5)
    c[0].metric("Records", s["records"])
    c[1].metric(f"Accuracy (reference, n={s['n_reference']})", fmt(s["accuracy_reference"], "%"))
    c[2].metric(f"LLM-assessed (n={s['n_llm_assessed']})", fmt(s["accuracy_llm_assessed"], "%"),
                help="Model grading its own answers: indicative only.")
    c[3].metric("Disagreement rate", fmt(s["disagreement_rate"], "%"))
    c[4].metric("KB grounded", fmt(s["kb_grounded_rate"], "%"))
    c = st.columns(2)
    c[0].metric("Median latency", fmt(s["latency_median_ms"], " ms"))
    c[1].metric("Mean latency", fmt(s["latency_mean_ms"], " ms"))
    by_country = tu.accuracy_by(rows, "Country/Region")
    if by_country:
        st.write("Accuracy by country (reference-graded rows)")
        st.dataframe(pd.DataFrame(by_country).T)
    by_level = tu.accuracy_by(rows, "Resource Level")
    if by_level:
        st.write("Accuracy by language resource level")
        st.dataframe(pd.DataFrame(by_level).T)
    mt = tu.mean_timings(timings or [])
    if mt:
        st.write("Mean time per stage (ms)")
        st.bar_chart(pd.Series(mt))


def show_table(rows, key, filename):
    df = pd.DataFrame(rows, columns=tu.TABLE_COLUMNS)
    st.dataframe(df)
    st.download_button("Download CSV (opens in Excel)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{filename}.csv", mime="text/csv", key=f"csv_{key}")
    try:
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button("Download Excel", buffer.getvalue(), file_name=f"{filename}.xlsx", key=f"xlsx_{key}")
    except Exception:
        pass


tab_single, tab_batch, tab_saved = st.tabs(["Ask one question", "Batch: fill the table", "Saved runs"])

# ------------------------------------------------------------------ single question
with tab_single:
    with st.form("single"):
        question = st.text_input("Question", "What is the cultural significance of Chuseok?")
        c1, c2, c3 = st.columns(3)
        country = c1.selectbox("Country", config.COUNTRIES)
        reference = c2.text_input("Expected answer (optional)", help="Use || between alternatives")
        category = c3.selectbox("Category (optional)", ["(auto)"] + tu.CATEGORIES)
        go = st.form_submit_button("Analyze")
    if go:
        item = {"question": question, "country": country,
                "reference": tu.to_answer_list(reference) if reference else None,
                "category": None if category == "(auto)" else category,
                "question_id": f"single-{int(time.time())}", "answer_type": None}
        with st.spinner("Generating both answers, comparing, retrieving, judging, grading..."):
            try:
                out = call_record(item, "web")
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        row = out["row"]
        c1, c2 = st.columns(2)
        c1.subheader(f"Local-language answer ({row['Local Language']})")
        c1.write(row["Local Answer"])
        c2.subheader("English answer")
        c2.write(row["English Answer"])
        m = st.columns(5)
        m[0].metric("Similarity", row["Similarity"] if row["Similarity"] is not None else "n/a")
        m[1].metric("Agreement", row["Agreement"])
        m[2].metric("Judge decision", row["Judge Decision"])
        m[3].metric("Final correct", row["Final Correct"], help=row["Correctness Basis"])
        m[4].metric("Latency", f"{row['Latency (ms)'] / 1000:.1f} s")
        if row["Retrieved Evidence"] != "-":
            st.subheader(f"Retrieved evidence (relevance {row['Evidence Relevance']}, KB grounded: {row['KB Grounded']})")
            st.text(row["Retrieved Evidence"])
        st.subheader("Final answer")
        st.success(row["Final Answer"])
        st.write("Table row")
        st.dataframe(pd.DataFrame([row], columns=tu.TABLE_COLUMNS))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Give many questions at once. Each one becomes a row of the evaluation table and is saved to the database.")
    source = st.radio("Questions from", ["Type / paste", "Upload file (CSV / JSONL / TXT)", "BLEnD sample"], horizontal=True)
    default_country = st.selectbox("Default country (when a line or row has none)", config.COUNTRIES, key="dc")
    run_id = st.text_input("Run name", f"web_{time.strftime('%Y%m%d_%H%M')}")
    items = []
    if source == "Type / paste":
        st.caption("One per line:  question | expected answer | category   (the last two are optional;  "
                   "start a line with a country name and | to override the default country).  "
                   "For multiple choice write the options in the question with \\n between them, e.g.  "
                   "What do people eat?\\nA. kimchi\\nB. pizza | A")
        text = st.text_area("Questions", height=180, placeholder=(
            "What is the traditional Korean clothing? | hanbok\n"
            "Ethiopia | What is the staple flatbread of Ethiopia? | injera | Food"))
        items = tu.parse_questions_text(text, default_country) if text.strip() else []
    elif source == "Upload file (CSV / JSONL / TXT)":
        up = st.file_uploader("File", type=["csv", "jsonl", "json", "txt"])
        if up:
            items = tu.parse_upload(up.name, up.getvalue(), default_country)
            st.caption("CSV columns: question, country, ground_truth (or reference/answer), category.  "
                       "eval_data/curated_questions.jsonl works as is.")
    else:
        c1, c2, c3 = st.columns(3)
        qtype = c1.selectbox("Type", ["saq", "mcq"])
        per_country = c2.number_input("Questions per country", 1, 200, 10)
        skip = c3.number_input("Skip first N per country", 0, 500, 20)
        if st.button("Load BLEnD questions"):
            try:
                from data_loader import load_samples
                samples = load_samples(qtype, int(per_country), int(skip))
                st.session_state["blend_items"] = [
                    tu._item(s["question"], s["country"], s["ground_truth"], None, s["question_id"], s["answer_type"])
                    for s in samples]
            except Exception as exc:
                st.error(f"Could not load BLEnD: {exc}")
        items = st.session_state.get("blend_items", [])
    if items:
        st.info(f"{len(items)} questions ready. About {len(items) * 15 // 60 + 1} minutes at ~15 s each "
                f"(depends on your API limits).")
        if st.button("Run all and fill the table", type="primary"):
            st.session_state["rows"], st.session_state["timings"] = [], []
            bar, live, failed = st.progress(0.0), st.empty(), []
            for i, item in enumerate(items, 1):
                try:
                    out = call_record(item, run_id)
                    st.session_state["rows"].append(out["row"])
                    st.session_state["timings"].append(out["timings"])
                except Exception as exc:
                    failed.append(f"{item['question'][:60]}: {exc}")
                bar.progress(i / len(items))
                live.dataframe(pd.DataFrame(st.session_state["rows"], columns=tu.TABLE_COLUMNS))
            if failed:
                st.warning(f"{len(failed)} failed (usually rate limits). Run again to retry those.")
                st.write(failed)
    if st.session_state.get("rows"):
        st.subheader("Evaluation table")
        show_summary(st.session_state["rows"], st.session_state.get("timings"))
        show_table(st.session_state["rows"], "batch", run_id)

# ------------------------------------------------------------------ saved runs
with tab_saved:
    try:
        runs = requests.get(f"{api_url}/runs", timeout=30).json()["runs"]
    except Exception as exc:
        runs = []
        st.error(f"Could not reach the API: {exc}")
    if runs:
        chosen = st.selectbox("Saved run", runs)
        if st.button("Load run"):
            data = requests.get(f"{api_url}/records", params={"run_id": chosen}, timeout=60).json()
            st.session_state["saved_rows"] = data["rows"]
        if st.session_state.get("saved_rows"):
            show_summary(st.session_state["saved_rows"])
            show_table(st.session_state["saved_rows"], "saved", chosen)
