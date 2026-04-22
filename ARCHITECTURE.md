# Architecture — LLM Data Platform

## Overview

An AI-powered real-time data platform that combines Kafka + Spark Structured Streaming with Claude LLM for autonomous anomaly detection, a RAG-based data catalog, and a LangChain agent for self-healing data quality.

```
Kafka (transactions-raw)
        │
        ▼
Spark Structured Streaming
  ├── Schema validation + DQ flags (null user, extreme amount, invalid currency, future timestamp)
  ├── Clean events  ──────────────────────► Delta Lake: transactions_clean   (partitioned by date)
  └── Anomalous events ────────────────────► Delta Lake: transactions_anomalies
                                                   │
                                                   ▼
                                        LLM Anomaly Detector (Claude claude-3-5-sonnet)
                                          - Batches up to 20 anomalies per API call
                                          - Root cause, business impact, remediation
                                          - Executive summary for stakeholders
                                          - Saves enriched report → Delta Lake: anomaly_reports
                                                   │
                                                   ▼
                                        LangChain ReAct Agent (DQ Agent)
                                          - check_pipeline_freshness tool
                                          - get_anomaly_summary tool
                                          - generate_remediation_sql tool
                                          - create_incident_report tool
                                          - Autonomous: runs full DQ check, decides next action

ChromaDB (persistent, local)
  ◄── catalog/embedder.py indexes Delta table metadata
  ──► RAG Data Catalog (Claude + semantic search)
        - Natural language queries over table catalog
        - "Which table has PII?" → retrieves top-K tables → LLM generates answer

FastAPI  ──► REST endpoints for catalog queries and anomaly reports
Airflow  ──► Orchestrates daily LLM analysis runs
Docker Compose ──► Local dev: Kafka, Spark, ChromaDB, FastAPI
```

## Key Design Decisions

**Why two Delta tables (clean + anomalies) instead of one flagged table?**  
Clean events need sub-second append throughput; anomaly records need to be held for async LLM processing. Separating them avoids the LLM batch job blocking the streaming path.

**Why batch anomalies to Claude instead of streaming?**  
LLM API calls have latency (1–3s) and cost per token. Batching 20 records per call reduces cost ~20x and allows the detector to identify cross-record patterns (e.g., "all from same country") that single-record calls would miss.

**Why LangChain ReAct agent instead of a fixed pipeline?**  
DQ remediation is conditional — different anomaly counts and freshness states require different actions. The ReAct loop lets the agent decide whether to generate SQL, file an incident, or both, without hardcoding every branch.

**Why ChromaDB for the catalog instead of a SQL metadata store?**  
The catalog use case is semantic search ("which table has fraud risk scores?"), not exact lookup. Vector similarity returns useful results even when the user doesn't know exact table names.

## Data Flow Latency

| Stage | Target latency |
|---|---|
| Kafka → Delta (clean) | < 15 seconds (trigger interval) |
| Anomaly → LLM report | < 5 minutes (Airflow batch) |
| Catalog query (RAG) | < 3 seconds |

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Apache Kafka |
| Processing | PySpark Structured Streaming |
| Storage | Delta Lake on AWS S3 |
| AI / LLM | Anthropic Claude (claude-3-5-sonnet) |
| Vector store | ChromaDB |
| Agent framework | LangChain ReAct |
| API | FastAPI |
| Orchestration | Apache Airflow |
| Infra | Docker Compose |

## Local Development

```bash
docker-compose up -d          # starts Kafka, Spark, ChromaDB, FastAPI
python pipeline/kafka_producer.py   # seed test events
python pipeline/spark_stream_processor.py
python ai/anomaly_detector.py $(date +%F)
```
