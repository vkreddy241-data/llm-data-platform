"""
LangChain Data Quality Agent: An autonomous AI agent that can:
  1. Query Delta Lake tables for quality metrics
  2. Call the LLM anomaly detector
  3. Generate and execute remediation SQL
  4. File incident reports
  5. Notify stakeholders

Uses Claude + LangChain's ReAct agent pattern with custom tools.
"""

import json
import logging
import os
from datetime import datetime, timezone

from anthropic import Anthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DELTA_CLEAN     = os.getenv("DELTA_CLEAN",     "s3a://vkreddy-llm-platform/delta/transactions_clean")
DELTA_ANOMALIES = os.getenv("DELTA_ANOMALIES", "s3a://vkreddy-llm-platform/delta/transactions_anomalies")

# Shared Spark session for tools
_spark = None


def get_spark() -> SparkSession:
    global _spark
    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("DQAgent")
            .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate()
        )
    return _spark


# ---------------------------------------------------------------------------
# LangChain Tools (each tool is an action the agent can take)
# ---------------------------------------------------------------------------

@tool
def get_table_stats(table_name: str) -> str:
    """Get row count, null rates, and basic stats for a Delta Lake table.
    Input: table name (transactions_clean or transactions_anomalies)"""
    paths = {
        "transactions_clean":     DELTA_CLEAN,
        "transactions_anomalies": DELTA_ANOMALIES,
    }
    if table_name not in paths:
        return f"Unknown table: {table_name}. Available: {list(paths.keys())}"

    spark = get_spark()
    from pyspark.sql import functions as F
    df = spark.read.format("delta").load(paths[table_name])
    total = df.count()
    null_rates = {}
    for col in df.columns[:10]:
        nulls = df.filter(F.col(col).isNull()).count()
        null_rates[col] = round(nulls / total * 100, 2) if total > 0 else 0

    return json.dumps({
        "table":      table_name,
        "row_count":  total,
        "null_rates_pct": null_rates,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    })


@tool
def get_anomaly_summary(run_date: str) -> str:
    """Get a summary of anomalies detected for a given run_date (YYYY-MM-DD).
    Input: run_date as YYYY-MM-DD string"""
    spark = get_spark()
    from pyspark.sql import functions as F
    df = spark.read.format("delta").load(DELTA_ANOMALIES)
    day_df = df.filter(F.col("date_part") == run_date)
    total = day_df.count()
    if total == 0:
        return json.dumps({"run_date": run_date, "total_anomalies": 0})

    by_type = day_df.groupBy("_anomaly_type").count().collect()
    return json.dumps({
        "run_date":        run_date,
        "total_anomalies": total,
        "by_type":         {r._anomaly_type: r["count"] for r in by_type},
    })


@tool
def check_pipeline_freshness(table_name: str) -> str:
    """Check how recently a Delta table was updated.
    Input: table name (transactions_clean or transactions_anomalies)"""
    paths = {
        "transactions_clean":     DELTA_CLEAN,
        "transactions_anomalies": DELTA_ANOMALIES,
    }
    if table_name not in paths:
        return f"Unknown table: {table_name}"

    spark = get_spark()
    from delta.tables import DeltaTable
    dt = DeltaTable.forPath(spark, paths[table_name])
    last_op = dt.history(1).select("timestamp", "operation").first()
    last_ts  = last_op["timestamp"]
    age_mins = (datetime.now(timezone.utc) - last_ts.replace(tzinfo=timezone.utc)).seconds // 60
    status   = "fresh" if age_mins < 30 else "stale"

    return json.dumps({
        "table":           table_name,
        "last_updated":    str(last_ts),
        "age_minutes":     age_mins,
        "freshness_status": status,
    })


@tool
def generate_remediation_sql(issue_description: str) -> str:
    """Generate SQL to remediate a specific data quality issue.
    Input: natural language description of the issue"""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Generate Delta Lake SQL to fix this data quality issue:
{issue_description}

Tables available:
- transactions_clean (path: {DELTA_CLEAN})
- transactions_anomalies (path: {DELTA_ANOMALIES})

Return ONLY the SQL statement(s), no explanation.""",
        }],
    )
    return response.content[0].text.strip()


@tool
def create_incident_report(summary: str) -> str:
    """Create a structured incident report for a data quality issue.
    Input: summary of the issue"""
    report = {
        "incident_id":   f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "severity":      "medium",
        "summary":       summary,
        "assigned_to":   "data-engineering-team",
        "status":        "open",
        "pipeline":      "llm-data-platform",
    }
    logger.info(f"Incident created: {report['incident_id']}")
    return json.dumps(report)


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

AGENT_PROMPT = PromptTemplate.from_template("""You are an autonomous Data Quality Agent for a real-time data pipeline.

Your job is to:
1. Monitor pipeline health and data quality
2. Identify and diagnose issues
3. Suggest or generate remediations
4. Create incident reports when needed

You have access to these tools:
{tools}

Tool names: {tool_names}

Use this format:
Thought: think about what to do
Action: tool_name
Action Input: input to the tool
Observation: result
... (repeat as needed)
Final Answer: your complete analysis and recommendations

Question: {input}
{agent_scratchpad}""")


def build_agent() -> AgentExecutor:
    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0,
    )
    tools = [
        get_table_stats,
        get_anomaly_summary,
        check_pipeline_freshness,
        generate_remediation_sql,
        create_incident_report,
    ]
    agent = create_react_agent(llm, tools, AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8)


def run_dq_check(run_date: str) -> str:
    agent_executor = build_agent()
    task = f"""
    Perform a complete data quality check for run_date={run_date}:
    1. Check freshness of the transactions_clean table
    2. Get the anomaly summary for {run_date}
    3. If anomalies > 50, get table stats and generate remediation SQL
    4. If pipeline is stale (age > 30 min), create an incident report
    5. Provide a final health assessment: HEALTHY / DEGRADED / CRITICAL
    """
    result = agent_executor.invoke({"input": task})
    return result["output"]


if __name__ == "__main__":
    today = datetime.now(timezone.utc).date().isoformat()
    output = run_dq_check(today)
    print(output)
