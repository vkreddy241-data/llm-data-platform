"""
LLM Schema Analyzer: Monitors Delta Lake tables for schema drift,
uses Claude to explain changes and assess impact on downstream consumers.

Runs daily. Compares today's schema vs baseline (stored in Delta log history).
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import anthropic
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DELTA_CLEAN     = os.getenv("DELTA_CLEAN", "s3a://vkreddy-llm-platform/delta/transactions_clean")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("LLMSchemaAnalyzer")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def get_schema_history(spark: SparkSession, delta_path: str) -> list[dict]:
    """Get schema from last 7 Delta versions."""
    dt = DeltaTable.forPath(spark, delta_path)
    history = dt.history(7).select("version", "timestamp", "operationMetrics").collect()
    schemas = []
    for row in history:
        try:
            df = spark.read.format("delta").option("versionAsOf", row.version).load(delta_path)
            schemas.append({
                "version":   int(row.version),
                "timestamp": str(row.timestamp),
                "columns":   [{"name": f.name, "type": str(f.dataType), "nullable": f.nullable}
                              for f in df.schema.fields],
            })
        except Exception as e:
            logger.warning(f"Could not read version {row.version}: {e}")
    return schemas


def detect_drift(schema_history: list[dict]) -> Optional[dict]:
    """Detect added, removed, or type-changed columns between latest and previous."""
    if len(schema_history) < 2:
        return None

    latest  = {c["name"]: c for c in schema_history[0]["columns"]}
    prev    = {c["name"]: c for c in schema_history[1]["columns"]}

    added   = [c for c in latest if c not in prev]
    removed = [c for c in prev if c not in latest]
    changed = [c for c in latest if c in prev and latest[c]["type"] != prev[c]["type"]]

    if not (added or removed or changed):
        return None

    return {
        "from_version":  schema_history[1]["version"],
        "to_version":    schema_history[0]["version"],
        "from_timestamp": schema_history[1]["timestamp"],
        "to_timestamp":   schema_history[0]["timestamp"],
        "added_columns":   added,
        "removed_columns": removed,
        "type_changes":    changed,
    }


def build_schema_prompt(drift: dict, schema_history: list[dict]) -> str:
    return f"""You are a Senior Data Engineer reviewing schema changes in a production Delta Lake table.

SCHEMA DRIFT DETECTED:
{json.dumps(drift, indent=2)}

FULL SCHEMA HISTORY (last {len(schema_history)} versions):
{json.dumps(schema_history, indent=2)}

Analyse this schema change and provide:
1. IMPACT ASSESSMENT: Which downstream consumers (BI reports, ML models, APIs) are likely affected?
2. BREAKING CHANGE: Is this a breaking change? Why/why not?
3. ROOT CAUSE HYPOTHESIS: What likely caused this drift (new data source, producer bug, intentional migration)?
4. RECOMMENDED ACTIONS: Step-by-step what the data team should do NOW
5. ROLLBACK PLAN: How to safely rollback using Delta time-travel if needed

Respond in JSON:
{{
  "is_breaking_change": true/false,
  "severity": "low|medium|high|critical",
  "impact_assessment": "string",
  "root_cause_hypothesis": "string",
  "recommended_actions": ["step1", "step2"],
  "rollback_sql": "string — Delta time-travel SQL if needed",
  "stakeholder_message": "2 sentence message to send to data consumers"
}}"""


def analyze_schema(delta_path: str) -> dict:
    spark  = build_spark()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    logger.info(f"Fetching schema history for {delta_path} ...")
    schema_history = get_schema_history(spark, delta_path)

    drift = detect_drift(schema_history)
    if drift is None:
        logger.info("No schema drift detected.")
        spark.stop()
        return {"status": "no_drift", "checked_at": datetime.now(timezone.utc).isoformat()}

    logger.info(f"Schema drift detected: +{len(drift['added_columns'])} cols, "
                f"-{len(drift['removed_columns'])} cols, ~{len(drift['type_changes'])} changed")

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": build_schema_prompt(drift, schema_history)}],
    )

    text  = response.content[0].text
    start = text.find("{")
    end   = text.rfind("}") + 1
    result = json.loads(text[start:end])
    result["drift_details"] = drift
    result["analyzed_at"]   = datetime.now(timezone.utc).isoformat()

    logger.info(f"Schema analysis: severity={result.get('severity')}, "
                f"breaking={result.get('is_breaking_change')}")

    spark.stop()
    return result


if __name__ == "__main__":
    result = analyze_schema(DELTA_CLEAN)
    print(json.dumps(result, indent=2))
