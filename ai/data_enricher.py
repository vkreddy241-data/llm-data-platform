"""
LLM Data Enricher: Uses Claude to enrich raw transaction records with
inferred attributes — sentiment, fraud risk score, product category
inference, and customer intent classification.

This replaces brittle regex rules with flexible LLM reasoning.
Runs as a Spark UDF so it scales across the cluster.
"""

import json
import logging
import os
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-3-haiku-20240307"   # fast + cheap for enrichment


def build_enrichment_prompt(batch: list[dict]) -> str:
    return f"""You are a data enrichment AI for an e-commerce platform.

For each transaction record below, infer and add:
1. "fraud_risk": "low" | "medium" | "high"  (based on amount, country, payment_method patterns)
2. "customer_intent": "impulse_buy" | "planned_purchase" | "subscription" | "refund_request"
3. "product_category": inferred category from product_id pattern (e.g. "electronics", "clothing", "unknown")
4. "geo_risk_flag": true if country is in high-fraud regions (e.g. unusual combos)
5. "enrichment_confidence": 0.0 to 1.0

INPUT RECORDS:
{json.dumps(batch, indent=2, default=str)}

Respond ONLY with a JSON array — one enrichment object per record, in the same order:
[
  {{
    "event_id": "string",
    "fraud_risk": "low|medium|high",
    "customer_intent": "string",
    "product_category": "string",
    "geo_risk_flag": true/false,
    "enrichment_confidence": float,
    "reasoning": "1 sentence"
  }},
  ...
]"""


def enrich_batch(records: list[dict]) -> list[dict]:
    """Enrich a batch of records using Claude."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": build_enrichment_prompt(records)}],
        )
        text  = response.content[0].text
        start = text.find("[")
        end   = text.rfind("]") + 1
        enrichments = json.loads(text[start:end])

        # Merge enrichments back into original records
        enrichment_map = {e["event_id"]: e for e in enrichments}
        for record in records:
            enriched = enrichment_map.get(record.get("event_id"), {})
            record.update({
                "fraud_risk":             enriched.get("fraud_risk", "unknown"),
                "customer_intent":        enriched.get("customer_intent", "unknown"),
                "product_category":       enriched.get("product_category", "unknown"),
                "geo_risk_flag":          enriched.get("geo_risk_flag", False),
                "enrichment_confidence":  enriched.get("enrichment_confidence", 0.0),
                "enrichment_reasoning":   enriched.get("reasoning", ""),
            })
        return records

    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        for record in records:
            record["fraud_risk"]  = "unknown"
            record["enrichment_confidence"] = 0.0
        return records


def enrich_spark_partition(iterator, batch_size: int = 25):
    """
    Spark mapPartitions function — batches rows and calls LLM once per batch.
    Usage: df.rdd.mapPartitions(enrich_spark_partition)
    """
    batch = []
    for row in iterator:
        batch.append(row.asDict())
        if len(batch) >= batch_size:
            yield from enrich_batch(batch)
            batch = []
    if batch:
        yield from enrich_batch(batch)


# ---------------------------------------------------------------------------
# Standalone enrichment for a single record (useful for API endpoint)
# ---------------------------------------------------------------------------
def enrich_single(record: dict) -> Optional[dict]:
    results = enrich_batch([record])
    return results[0] if results else None


if __name__ == "__main__":
    # Quick test with a sample record
    sample = {
        "event_id":       "EVT-123456",
        "user_id":        "USR-5001",
        "product_id":     "PRD-101",
        "amount":         299.99,
        "currency":       "USD",
        "event_type":     "purchase",
        "country":        "US",
        "payment_method": "crypto",
    }
    result = enrich_single(sample)
    print(json.dumps(result, indent=2))
