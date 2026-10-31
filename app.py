"""FastAPI service: the ONLY server for the project (replaces main.py).

    uvicorn app:app --host 127.0.0.1 --port 8000
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
import database
from disagreement import get_embedding_model
from kb_retriever import load_retriever_resources, search_kb
from llm import LLMError
from pipeline import run_pipeline
from records import build_record, rows_for_run
from textutils import is_correct


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    load_retriever_resources()      # models + index are loaded once, then stay in memory
    get_embedding_model()
    yield


app = FastAPI(title="Dual-Language Cultural QA API", version="2.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    country: Optional[str] = ""
    language: Optional[str] = "English"
    mode: Optional[str] = "dual_kb"
    answer_format: Optional[str] = "free"        # "free" or "mcq"
    local_question: Optional[str] = None
    question_id: Optional[str] = None
    ground_truth: Optional[str] = None


class RecordRequest(BaseModel):
    question: str
    country: str
    language: Optional[str] = None
    mode: Optional[str] = "dual_kb"
    answer_type: Optional[str] = None            # "mcq" / "saq"; auto-detected if empty
    reference: Optional[str] = None              # expected answer(s); 'a||b' for several, option letter for MCQ
    category: Optional[str] = None               # auto-classified if empty
    question_id: Optional[str] = None
    run_id: Optional[str] = "web"
    show_evidence: Optional[bool] = True
    assess_without_reference: Optional[bool] = True
    no_cache: Optional[bool] = False             # true latency: bypass the LLM cache for this call
    local_question: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str
    country: Optional[str] = None
    top_k: Optional[int] = config.TOP_K


@app.get("/")
def root():
    return {"status": "ok", "modes": config.MODES, "countries": config.COUNTRIES}


@app.post("/record")
def record_endpoint(request: RecordRequest):
    """One fully filled evaluation-table row (pipeline + grading), saved to the database."""
    previous = config.USE_CACHE
    if request.no_cache:
        config.USE_CACHE = False
    try:
        out = build_record(
            question=request.question, country=request.country, language=request.language,
            mode=request.mode or "dual_kb", answer_type=request.answer_type,
            reference=request.reference, category=request.category,
            question_id=request.question_id, run_id=request.run_id or "web",
            show_evidence=bool(request.show_evidence),
            assess_without_reference=bool(request.assess_without_reference),
            local_question=request.local_question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        config.USE_CACHE = previous
    result = out["result"]
    return {"row": out["row"], "timings": out["timings"], "local_answer_en": result["local_answer_en"]}


@app.get("/runs")
def runs_endpoint():
    return {"runs": database.list_run_ids()}


@app.get("/records")
def records_endpoint(run_id: str):
    rows = rows_for_run(run_id)
    return {"run_id": run_id, "count": len(rows), "rows": rows}


@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    results = search_kb(query=request.query, country=request.country, top_k=request.top_k)
    return {"query": request.query, "country": request.country,
            "count": len(results), "results": results}


@app.post("/process_cultural_qa")
@app.post("/api/evaluate")
def evaluate_endpoint(request: QueryRequest):
    try:
        result = run_pipeline(
            question=request.question, country=request.country or "",
            language=request.language or "English", mode=request.mode or "dual_kb",
            answer_format=request.answer_format or "free",
            local_question=request.local_question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    answer_type = "mcq" if request.answer_format == "mcq" else "saq"
    correct = None
    if request.ground_truth:
        correct = int(is_correct(result["final_answer_en"], request.ground_truth, answer_type))

    database.log_run({
        "run_id": "api", "mode": result["mode"], "question_id": request.question_id,
        "country": request.country, "language": result["language"],
        "answer_type": answer_type, "question": request.question,
        "ground_truth": request.ground_truth, "local_answer": result["local_answer"],
        "local_answer_en": result["local_answer_en"], "english_answer": result["english_answer"],
        "similarity": result["similarity_score"], "is_disagreed": result["is_disagreed"],
        "used_judge": result["used_judge"], "retrieved_facts": result["retrieved_facts"],
        "final_answer": result["final_answer"], "is_correct": correct,
        "latency_ms": result["latency_ms"], "agreement": result["agreement"],
        "judge_decision": result["judge_decision"], "evidence_relevance": result["evidence_relevance"],
        "evidence_used": int(bool(result["retrieved_facts"])), "timings": result["timings"],
    })
    result["question_id"] = request.question_id
    result["is_correct"] = correct
    return result
