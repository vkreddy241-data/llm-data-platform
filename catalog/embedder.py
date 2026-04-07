"""
Data Catalog Embedder: Scans Delta Lake tables, extracts metadata
(schema, sample rows, column stats), generates natural language descriptions
using Claude, then embeds them into ChromaDB for semantic search.

Run once to bootstrap the catalog, then daily to keep it fresh.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL    = "claude-3-5-sonnet-20241022"
CHROMA_PATH        = os.getenv("CHROMA_PATH", "./chroma_catalog")
EMBEDDING_MODEL    = "text-embedding-3-small"   # OpenAI embeddings via ChromaDB
COLLECTION_NAME    = "data_catalog"

DELTA_TABLES = {
    "transactions_clean":     "s3a://vkreddy-llm-platform/delta/transactions_clean",
    "transactions_anomalies": "s3a://vkreddy-llm-platform/delta/transactions_anomalies",
    "anomaly_reports":        "s3a://vkreddy-llm-platform/delta/anomaly_reports",
}


@dataclass
class TableMetadata:
    table_name: str
    delta_path: str
    row_count: int
    column_count: int
    columns: list[dict]
    sample_values: dict
    stats: dict
    llm_description: Optional[str] = None


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DataCatalogEmbedder")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def extract_table_metadata(spark: SparkSession, table_name: str, path: str) -> TableMetadata:
    df = spark.read.format("delta").load(path)
    row_count = df.count()

    columns = [
        {"name": f.name, "type": str(f.dataType), "nullable": f.nullable}
        for f in df.schema.fields
    ]

    # Sample values per column
    sample = df.limit(5).toPandas()
    sample_values = {col: sample[col].tolist() for col in sample.columns}

    # Basic stats for numeric columns
    numeric_cols = [f.name for f in df.schema.fields
                    if "Double" in str(f.dataType) or "Long" in str(f.dataType)]
    stats = {}
    if numeric_cols:
        stats_df = df.select([
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
            F.avg(c).alias(f"{c}_avg"),
        ] for c in numeric_cols[:5])
        stats = stats_df.first().asDict() if stats_df.count() > 0 else {}

    return TableMetadata(
        table_name=table_name,
        delta_path=path,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        sample_values=sample_values,
        stats=stats,
    )


def generate_llm_description(client: anthropic.Anthropic, metadata: TableMetadata) -> str:
    prompt = f"""You are a data catalog assistant. Given the following Delta Lake table metadata,
generate a comprehensive, human-readable description for a data catalog.

TABLE: {metadata.table_name}
PATH: {metadata.delta_path}
ROW COUNT: {metadata.row_count:,}
COLUMNS ({metadata.column_count}): {json.dumps(metadata.columns, indent=2)}
SAMPLE VALUES: {json.dumps({k: v[:3] for k, v in metadata.sample_values.items()}, indent=2, default=str)}

Write a 3-5 sentence description covering:
1. What this table contains and its business purpose
2. Key columns and their meaning
3. How it fits in the data pipeline (bronze/silver/gold layer)
4. Who the typical consumers of this table are (BI analysts, ML engineers, etc.)
5. Any data quality or freshness SLAs

Be specific, technical, and helpful for someone discovering this table for the first time."""

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def embed_to_chroma(metadata: TableMetadata, collection) -> None:
    doc_text = f"""
TABLE: {metadata.table_name}
DESCRIPTION: {metadata.llm_description}
COLUMNS: {', '.join(c['name'] for c in metadata.columns)}
ROW COUNT: {metadata.row_count:,}
PATH: {metadata.delta_path}
"""
    collection.upsert(
        documents=[doc_text],
        ids=[metadata.table_name],
        metadatas=[{
            "table_name":   metadata.table_name,
            "delta_path":   metadata.delta_path,
            "row_count":    str(metadata.row_count),
            "column_count": str(metadata.column_count),
            "columns_json": json.dumps(metadata.columns),
        }],
    )
    logger.info(f"Embedded {metadata.table_name} into ChromaDB.")


def build_catalog():
    spark  = build_spark()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )
    collection = chroma.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)

    for table_name, path in DELTA_TABLES.items():
        logger.info(f"Processing {table_name} ...")
        metadata = extract_table_metadata(spark, table_name, path)
        metadata.llm_description = generate_llm_description(client, metadata)
        embed_to_chroma(metadata, collection)

    spark.stop()
    logger.info(f"Catalog built: {len(DELTA_TABLES)} tables embedded.")


if __name__ == "__main__":
    build_catalog()
