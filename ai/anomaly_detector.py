"""
LLM Anomaly Detector: Reads flagged anomalies from Delta Lake,
sends batches to Claude (Anthropic API) for natural language explanation,
root cause analysis, and remediation suggestions.

Output: enriched anomaly report saved back to Delta Lake + Slack/email alert.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import anthropic
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DELTA_ANOMALIES  = os.getenv("DELTA_ANOMALIES", "s3a://vkreddy-llm-platform/delta/transactions_anomalies")
DELTA_REPORTS    = os.getenv("DELTA_REPORTS",   "s3a://vkreddy-llm-platform/delta/anomaly_reports")
ANTHROPIC_MODEL  = "claude-3-5-sonnet-20241022"
BATCH_SIZE       = 20   # anomalies per LLM call


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("LLMAnomalyDetector")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def load_recent_anomalies(spark: SparkSession, run_date: str):
    return (
        spark.read.format("delta").load(DELTA_ANOMALIES)
        .filter(F.col("date_part") == run_date)
        .filter(F.col("llm_analyzed").isNull())   # not yet analyzed
        .orderBy(F.col("anomaly_score").desc())
        .limit(200)
    )


def build_analysis_prompt(anomalies: list[dict]) -> str:
    return f"""You are a Senior Data Engineer analyzing data quality anomalies in a real-time e-commerce transaction pipeline.

I'm giving you {len(anomalies)} anomalous transaction records detected today. For each anomaly:
1. Identify the ROOT CAUSE of the data quality issue
2. Assess the BUSINESS IMPACT (low / medium / high / critical)
3. Suggest a specific REMEDIATION action (e.g. filter, impute, reject, alert upstream)
4. Identify any PATTERNS across multiple anomalies (e.g. all from same country, time cluster)

ANOMALY RECORDS:
{json.dumps(anomalies, indent=2, default=str)}

Respond with a structured JSON array, one object per anomaly_type group:
{{
  "anomaly_groups": [
    {{
      "anomaly_type": "string",
      "count": int,
      "root_cause": "string",
      "business_impact": "low|medium|high|critical",
      "remediation": "string",
      "example_event_ids": ["EVT-xxx", "EVT-yyy"],
      "pattern_insight": "string or null"
    }}
  ],
  "overall_pipeline_health": "healthy|degraded|critical",
  "executive_summary": "2-3 sentence summary for stakeholders",
  "recommended_actions": ["action1", "action2"]
}}"""


def call_llm(client: anthropic.Anthropic, anomalies: list[dict]) -> Optional[dict]:
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": build_analysis_prompt(anomalies),
            }],
        )
        text = response.content[0].text
        # Extract JSON from response
        start = text.find("{")
        end   = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def analyze_anomalies(run_date: str) -> dict:
    spark  = build_spark()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    df = load_recent_anomalies(spark, run_date)
    records = [row.asDict() for row in df.collect()]

    if not records:
        logger.info("No anomalies found for analysis.")
        return {"status": "no_anomalies"}

    logger.info(f"Analyzing {len(records)} anomalies with Claude ...")

    # Process in batches
    all_results = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i: i + BATCH_SIZE]
        result = call_llm(client, batch)
        if result:
            all_results.append(result)
            logger.info(f"Batch {i // BATCH_SIZE + 1}: pipeline_health={result.get('overall_pipeline_health')}")

    # Save reports back to Delta
    if all_results:
        reports_df = spark.createDataFrame([
            {
                "run_date":         run_date,
                "analyzed_at":      datetime.now(timezone.utc).isoformat(),
                "anomaly_count":    len(records),
                "pipeline_health":  all_results[0].get("overall_pipeline_health"),
                "executive_summary": all_results[0].get("executive_summary"),
                "full_report":      json.dumps(all_results[0]),
            }
        ])
        reports_df.write.format("delta").mode("append").save(DELTA_REPORTS)
        logger.info(f"Report saved → {DELTA_REPORTS}")

    spark.stop()
    return all_results[0] if all_results else {}


if __name__ == "__main__":
    import sys
    run_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).date().isoformat()
    result = analyze_anomalies(run_date)
    print(json.dumps(result, indent=2))
