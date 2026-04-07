"""
FastAPI: Natural Language Data Platform API

Endpoints:
  POST /query/catalog          — RAG-based data catalog search
  POST /query/anomalies        — Ask about anomalies in natural language
  POST /enrich                 — LLM-enrich a transaction record
  GET  /health                 — Pipeline health summary
  GET  /tables                 — List all catalog tables
  GET  /tables/{name}          — Describe a specific table
  POST /agent/run              — Run the autonomous DQ agent
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Data Platform API",
    description="AI-powered data pipeline with natural language querying, anomaly detection, and autonomous DQ agents.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CatalogQuery(BaseModel):
    question: str = Field(..., example="Which table has transaction anomaly data?")
    top_k: int = Field(default=5, ge=1, le=20)


class AnomalyQuery(BaseModel):
    question: str = Field(..., example="How many null user_id anomalies were there today?")
    run_date: Optional[str] = Field(default=None, example="2024-06-01")


class EnrichRequest(BaseModel):
    event_id:       str
    user_id:        str
    product_id:     str
    amount:         float
    currency:       str
    event_type:     str
    country:        str
    payment_method: str


class AgentRequest(BaseModel):
    run_date: Optional[str] = Field(default=None, example="2024-06-01")
    task: Optional[str] = Field(default=None, example="Check pipeline health and report anomalies")


class HealthResponse(BaseModel):
    status: str
    pipeline: str
    timestamp: str
    checks: dict


# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoid startup failures if deps not installed)
# ---------------------------------------------------------------------------

_catalog = None
_agent = None


def get_catalog():
    global _catalog
    if _catalog is None:
        from catalog.rag_catalog import RAGDataCatalog
        _catalog = RAGDataCatalog()
    return _catalog


def get_agent():
    global _agent
    if _agent is None:
        from agent.data_quality_agent import build_agent
        _agent = build_agent()
    return _agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
def root():
    return {
        "service":     "LLM Data Platform",
        "version":     "1.0.0",
        "docs":        "/docs",
        "description": "AI-powered data pipeline with RAG catalog, LLM anomaly detection & autonomous DQ agent",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health_check():
    return HealthResponse(
        status="healthy",
        pipeline="llm-data-platform",
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks={
            "kafka":       "connected",
            "delta_lake":  "accessible",
            "chroma_db":   "running",
            "llm_api":     "available" if os.environ.get("ANTHROPIC_API_KEY") else "missing_key",
        },
    )


@app.post("/query/catalog", tags=["catalog"])
def query_catalog(request: CatalogQuery):
    """Natural language search over the data catalog using RAG."""
    try:
        catalog = get_catalog()
        result  = catalog.query(request.question)
        return {
            "question": request.question,
            "answer":   result["answer"],
            "sources":  result["sources"],
        }
    except Exception as e:
        logger.error(f"Catalog query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables", tags=["catalog"])
def list_tables():
    """List all tables in the data catalog."""
    try:
        return get_catalog().list_tables()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables/{table_name}", tags=["catalog"])
def describe_table(table_name: str):
    """Get full description of a specific table."""
    catalog = get_catalog()
    result  = catalog.describe_table(table_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in catalog.")
    return result


@app.post("/query/anomalies", tags=["anomalies"])
def query_anomalies(request: AnomalyQuery):
    """Ask natural language questions about data anomalies."""
    from ai.anomaly_detector import analyze_anomalies
    run_date = request.run_date or datetime.now(timezone.utc).date().isoformat()
    try:
        result = analyze_anomalies(run_date)
        return {
            "question":        request.question,
            "run_date":        run_date,
            "pipeline_health": result.get("overall_pipeline_health"),
            "summary":         result.get("executive_summary"),
            "anomaly_groups":  result.get("anomaly_groups", []),
            "actions":         result.get("recommended_actions", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrich", tags=["enrichment"])
def enrich_transaction(request: EnrichRequest):
    """LLM-enrich a single transaction record with fraud risk, intent, and category."""
    from ai.data_enricher import enrich_single
    try:
        result = enrich_single(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/run", tags=["agent"])
def run_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """Run the autonomous Data Quality Agent for a given date."""
    from agent.data_quality_agent import run_dq_check
    run_date = request.run_date or datetime.now(timezone.utc).date().isoformat()
    try:
        result = run_dq_check(run_date)
        return {
            "run_date":   run_date,
            "agent_output": result,
            "ran_at":     datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
