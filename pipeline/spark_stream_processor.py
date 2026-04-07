"""
Spark Structured Streaming: Reads from Kafka, validates schema,
writes clean events to Delta Lake bronze, and flags anomalies
to a separate Delta table for LLM analysis.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
)

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP  = "localhost:9092"
KAFKA_TOPIC      = "transactions-raw"
DELTA_CLEAN      = "s3a://vkreddy-llm-platform/delta/transactions_clean"
DELTA_ANOMALIES  = "s3a://vkreddy-llm-platform/delta/transactions_anomalies"
CHECKPOINT_CLEAN = "s3a://vkreddy-llm-platform/checkpoints/clean"
CHECKPOINT_ANOM  = "s3a://vkreddy-llm-platform/checkpoints/anomalies"

EVENT_SCHEMA = StructType([
    StructField("event_id",       StringType(),    True),
    StructField("user_id",        StringType(),    True),
    StructField("product_id",     StringType(),    True),
    StructField("amount",         DoubleType(),    True),
    StructField("currency",       StringType(),    True),
    StructField("event_type",     StringType(),    True),
    StructField("timestamp",      StringType(),    True),
    StructField("country",        StringType(),    True),
    StructField("payment_method", StringType(),    True),
    StructField("_anomaly_type",  StringType(),    True),
])

VALID_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
MAX_AMOUNT = 10000.0


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("LLMDataPlatform-Streaming")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def parse_stream(spark: SparkSession):
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )
    return (
        raw
        .select(F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_ts",   F.to_timestamp("timestamp"))
        .withColumn("ingest_ts",  F.current_timestamp())
        .withColumn("date_part",  F.to_date("event_ts"))
    )


def add_quality_flags(df):
    return (
        df
        .withColumn("flag_null_user",
                    F.col("user_id").isNull())
        .withColumn("flag_extreme_amount",
                    F.col("amount") > MAX_AMOUNT)
        .withColumn("flag_negative_amount",
                    F.col("amount") < 0)
        .withColumn("flag_invalid_currency",
                    ~F.col("currency").isin(VALID_CURRENCIES))
        .withColumn("flag_future_ts",
                    F.col("event_ts") > F.current_timestamp())
        .withColumn("flag_schema_drift",
                    F.col("_anomaly_type") == "schema_drift")
        .withColumn("is_anomaly",
                    F.col("flag_null_user") |
                    F.col("flag_extreme_amount") |
                    F.col("flag_negative_amount") |
                    F.col("flag_invalid_currency") |
                    F.col("flag_future_ts") |
                    F.col("flag_schema_drift"))
        .withColumn("anomaly_score",
                    (F.col("flag_null_user").cast("int") +
                     F.col("flag_extreme_amount").cast("int") +
                     F.col("flag_negative_amount").cast("int") +
                     F.col("flag_invalid_currency").cast("int") +
                     F.col("flag_future_ts").cast("int")).cast("double"))
    )


def write_stream(df, path: str, checkpoint: str, filter_expr=None):
    out = df.filter(filter_expr) if filter_expr is not None else df
    return (
        out.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint)
        .option("mergeSchema", "true")
        .partitionBy("date_part")
        .trigger(processingTime="15 seconds")
        .start(path)
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    parsed = parse_stream(spark)
    flagged = add_quality_flags(parsed)

    # Stream 1: clean events
    write_stream(flagged, DELTA_CLEAN, CHECKPOINT_CLEAN,
                 filter_expr=F.col("is_anomaly") == False)   # noqa: E712

    # Stream 2: anomalies → LLM analyzer
    write_stream(flagged, DELTA_ANOMALIES, CHECKPOINT_ANOM,
                 filter_expr=F.col("is_anomaly") == True)    # noqa: E712

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
