# LLM Data Platform — AI-Powered Data Pipeline

![CI](https://github.com/vkreddy241-data/llm-data-platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Anthropic](https://img.shields.io/badge/Claude-3.5%20Sonnet-orange?logo=anthropic)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?logo=langchain)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-231F20?logo=apachekafka)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.0-00ADD8)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6F00)

A **production-grade AI-powered data platform** that combines real-time streaming pipelines with Large Language Models — featuring autonomous anomaly detection, natural language data catalog search (RAG), LLM-based data enrichment, and an agentic Data Quality agent built with LangChain + Claude.

> 🔥 Built to match 2025-2026 Data Engineer + AI Engineer job requirements: RAG · LLMs · Agents · Streaming · Delta Lake

---

## Architecture

```
Real-time Transactions (Kafka Producer)
             │
             ▼ Topic: transactions-raw
      ┌──────────────────────────────────┐
      │   Spark Structured Streaming     │
      │   - Schema validation            │
      │   - 6 anomaly flag rules         │
      │   - anomaly_score calculation    │
      └──────┬───────────────────────────┘
             │
    ┌────────┴──────────┐
    ▼                   ▼
Delta Lake           Delta Lake
(clean)             (anomalies)
    │                   │
    │         ┌─────────┘
    │         ▼
    │   🤖 LLM Anomaly Detector (Claude)
    │      - Root cause analysis
    │      - Business impact assessment
    │      - Remediation suggestions
    │      - Saves report → Delta Lake
    │
    ├──► 🔍 Schema Analyzer (Claude)
    │      - Detects schema drift via Delta history
    │      - Breaking change assessment
    │      - Rollback SQL generation
    │
    ├──► ✨ Data Enricher (Claude Haiku)
    │      - Fraud risk scoring
    │      - Customer intent classification
    │      - Geo risk flagging
    │      - Runs as Spark UDF (scales across cluster)
    │
    ├──► 🧠 LangChain DQ Agent (ReAct pattern)
    │      Tools: get_table_stats, get_anomaly_summary,
    │             check_freshness, generate_sql, create_incident
    │      Autonomous health checks + incident creation
    │
    └──► 📚 RAG Data Catalog (ChromaDB + Claude)
           - Claude generates table descriptions
           - ChromaDB stores embeddings
           - Natural language search: "which table has fraud data?"
                    │
                    ▼
          🌐 FastAPI (REST + /docs)
             /query/catalog  — RAG search
             /query/anomalies — NL anomaly queries
             /enrich          — LLM enrichment
             /agent/run       — Trigger DQ agent
             /health          — Pipeline status
```

## Key Features

| Feature | Tech | What It Does |
|---|---|---|
| **LLM Anomaly Detection** | Claude 3.5 Sonnet | Explains anomalies in plain English, assesses business impact |
| **Schema Drift Analysis** | Claude + Delta history | Detects breaking changes, generates rollback SQL |
| **Data Enrichment** | Claude Haiku (fast) | Adds fraud risk, customer intent, geo risk as Spark UDF |
| **RAG Data Catalog** | ChromaDB + Claude | Ask "which table has PII?" in natural language |
| **Autonomous DQ Agent** | LangChain + Claude | ReAct agent with 5 tools, auto-creates incident reports |
| **FastAPI** | FastAPI + Pydantic | REST API for all AI features with Swagger docs |
| **Real-time Pipeline** | Kafka + Spark Streaming | Sub-15s micro-batches into Delta Lake |
| **Orchestration** | Airflow | Hourly health checks + daily full analysis |

## Project Structure

```
llm-data-platform/
├── pipeline/
│   ├── kafka_producer.py          # Simulates transactions (with anomalies)
│   └── spark_stream_processor.py  # Streaming → Delta Lake (clean + anomalies)
├── ai/
│   ├── anomaly_detector.py        # Claude: root cause + impact analysis
│   ├── schema_analyzer.py         # Claude: schema drift detection + rollback SQL
│   └── data_enricher.py           # Claude Haiku: fraud risk + intent (Spark UDF)
├── catalog/
│   ├── embedder.py                # Claude → ChromaDB: embed table descriptions
│   └── rag_catalog.py             # RAG: natural language catalog search
├── agent/
│   └── data_quality_agent.py      # LangChain ReAct agent + 5 custom tools
├── api/
│   └── main.py                    # FastAPI: all features as REST endpoints
├── airflow/dags/
│   └── llm_platform_dag.py        # Hourly orchestration DAG
├── tests/
│   └── test_llm_platform.py       # 16 unit tests (no LLM calls needed)
├── docker-compose.yml             # Kafka + ChromaDB + API local stack
└── Dockerfile
```

## Quick Start

### 1. Set environment variables
```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. Start local stack
```bash
docker-compose up -d
# Kafka UI:  http://localhost:8080
# API docs:  http://localhost:8000/docs
# ChromaDB:  http://localhost:8001
```

### 3. Run the producer
```bash
pip install -r requirements.txt
python pipeline/kafka_producer.py
```

### 4. Query the catalog (natural language)
```bash
curl -X POST http://localhost:8000/query/catalog \
  -H "Content-Type: application/json" \
  -d '{"question": "Which table contains fraud risk scores?"}'
```

### 5. Enrich a transaction
```bash
curl -X POST http://localhost:8000/enrich \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "EVT-001", "user_id": "USR-1", "product_id": "PRD-1",
    "amount": 999.0, "currency": "USD", "event_type": "purchase",
    "country": "US", "payment_method": "crypto"
  }'
```

### 6. Run tests (no API keys needed)
```bash
pytest tests/ -v
```

## Sample LLM Outputs

**Anomaly Detection:**
```json
{
  "overall_pipeline_health": "degraded",
  "executive_summary": "18 null user_id anomalies detected, concentrated between 14:00-15:00 UTC, suggesting an upstream producer outage in the US region.",
  "anomaly_groups": [
    {
      "anomaly_type": "null_user",
      "count": 18,
      "root_cause": "Producer-side authentication failure causing user context loss",
      "business_impact": "high",
      "remediation": "Reject events with null user_id at Kafka consumer level; alert upstream team"
    }
  ]
}
```

**RAG Catalog Query:** `"Which table has anomaly data?"`
```
The transactions_anomalies Delta Lake table (s3a://vkreddy-llm-platform/delta/transactions_anomalies)
contains all flagged anomalous events. It includes 6 boolean quality flags, an anomaly_score column,
and is partitioned by date_part for efficient querying. Primary consumers are the LLM anomaly
detector and the Data Quality Agent.
```

## Tech Stack

**AI/LLM:** Anthropic Claude 3.5 Sonnet · Claude Haiku · LangChain 0.2 (ReAct agents)
**Vector DB:** ChromaDB 0.5 (local persistent store)
**Streaming:** Apache Kafka 3.5 · Spark Structured Streaming 3.5
**Storage:** Delta Lake 3.0 · AWS S3
**API:** FastAPI 0.111 · Pydantic v2
**Orchestration:** Apache Airflow 2.8
**Dev:** Docker Compose · GitHub Actions CI

---
Built by [Vikas Reddy Amaravathi](https://linkedin.com/in/vikas-reddy-a-avr03) — Azure Data Engineer @ Cigna
