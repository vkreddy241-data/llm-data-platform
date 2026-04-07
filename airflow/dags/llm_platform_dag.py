"""
Airflow DAG: Orchestrates the LLM Data Platform pipeline.

Schedule: Every hour (streaming health checks) + Daily (full analysis)
Flow:
  Check pipeline freshness
  → Run LLM Anomaly Detector
  → Run Schema Analyzer
  → Run Data Quality Agent
  → Refresh RAG Catalog
  → Notify on Slack/email if critical
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

DEFAULT_ARGS = {
    "owner":            "vikas-reddy",
    "depends_on_past":  False,
    "email":            ["vkreddy241@gmail.com"],
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def check_freshness(**context) -> str:
    """Check if streaming pipeline has written data in last 30 min."""
    run_date = context["ds"]
    # In production: check Delta table last modified timestamp
    print(f"Freshness check for {run_date} — OK")
    return "run_anomaly_detector"   # always proceed for demo; in prod: branch on stale


def run_anomaly_detector(**context):
    from ai.anomaly_detector import analyze_anomalies
    run_date = context["ds"]
    result = analyze_anomalies(run_date)
    health = result.get("overall_pipeline_health", "unknown")
    print(f"Pipeline health: {health}")
    context["ti"].xcom_push(key="pipeline_health", value=health)
    context["ti"].xcom_push(key="anomaly_summary",
                            value=result.get("executive_summary", ""))


def run_schema_analyzer(**context):
    import os
    from ai.schema_analyzer import analyze_schema
    delta_path = os.getenv("DELTA_CLEAN", "s3a://vkreddy-llm-platform/delta/transactions_clean")
    result = analyze_schema(delta_path)
    if result.get("is_breaking_change"):
        raise ValueError(
            f"BREAKING schema change detected! Severity: {result.get('severity')}\n"
            f"{result.get('stakeholder_message')}"
        )
    print(f"Schema check: {result.get('status', 'ok')}")


def run_dq_agent(**context):
    from agent.data_quality_agent import run_dq_check
    run_date = context["ds"]
    output = run_dq_check(run_date)
    print(f"DQ Agent output:\n{output}")


def refresh_catalog(**context):
    from catalog.embedder import build_catalog
    build_catalog()
    print("RAG catalog refreshed.")


def branch_on_health(**context):
    health = context["ti"].xcom_pull(task_ids="run_anomaly_detector", key="pipeline_health")
    if health == "critical":
        return "send_critical_alert"
    if health == "degraded":
        return "send_degraded_alert"
    return "pipeline_healthy"


def send_alert(severity: str, **context):
    summary = context["ti"].xcom_pull(task_ids="run_anomaly_detector", key="anomaly_summary")
    run_date = context["ds"]
    # In production: send to Slack via SlackWebhookOperator or PagerDuty
    print(f"[{severity.upper()}] ALERT for {run_date}: {summary}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="llm_data_platform",
    default_args=DEFAULT_ARGS,
    description="AI-powered data pipeline: LLM anomaly detection, schema analysis, RAG catalog",
    schedule_interval="0 * * * *",   # hourly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["llm", "ai", "data-quality", "rag", "anthropic"],
) as dag:

    freshness_check = BranchPythonOperator(
        task_id="check_pipeline_freshness",
        python_callable=check_freshness,
    )

    anomaly_task = PythonOperator(
        task_id="run_anomaly_detector",
        python_callable=run_anomaly_detector,
    )

    schema_task = PythonOperator(
        task_id="run_schema_analyzer",
        python_callable=run_schema_analyzer,
    )

    agent_task = PythonOperator(
        task_id="run_dq_agent",
        python_callable=run_dq_agent,
    )

    catalog_task = PythonOperator(
        task_id="refresh_rag_catalog",
        python_callable=refresh_catalog,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_health",
        python_callable=branch_on_health,
    )

    critical_alert = PythonOperator(
        task_id="send_critical_alert",
        python_callable=lambda **ctx: send_alert("critical", **ctx),
    )

    degraded_alert = PythonOperator(
        task_id="send_degraded_alert",
        python_callable=lambda **ctx: send_alert("degraded", **ctx),
    )

    healthy = EmptyOperator(task_id="pipeline_healthy")

    done = EmptyOperator(
        task_id="done",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    (
        freshness_check
        >> anomaly_task
        >> schema_task
        >> agent_task
        >> catalog_task
        >> branch_task
        >> [critical_alert, degraded_alert, healthy]
        >> done
    )
