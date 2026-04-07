"""
Kafka Producer: Simulates real-time e-commerce transaction events.
Intentionally injects anomalies (null fields, extreme amounts, schema drift)
so the LLM anomaly detector has something to catch.

Topic: transactions-raw
"""

import json
import random
import time
import logging
from datetime import datetime, timezone
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions-raw"
ANOMALY_RATE = 0.08   # 8% of events are anomalous


def normal_event() -> dict:
    return {
        "event_id":      f"EVT-{random.randint(100000, 999999)}",
        "user_id":       f"USR-{random.randint(1000, 9999)}",
        "product_id":    f"PRD-{random.randint(100, 999)}",
        "amount":        round(random.uniform(5.0, 500.0), 2),
        "currency":      random.choice(["USD", "EUR", "GBP"]),
        "event_type":    random.choice(["purchase", "refund", "cart_add"]),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "country":       random.choice(["US", "UK", "DE", "FR", "CA"]),
        "payment_method": random.choice(["card", "paypal", "crypto"]),
    }


def anomalous_event() -> dict:
    """Generates events with various data quality problems."""
    anomaly_type = random.choice([
        "null_user",
        "extreme_amount",
        "invalid_currency",
        "future_timestamp",
        "negative_amount",
        "schema_drift",
    ])
    event = normal_event()

    if anomaly_type == "null_user":
        event["user_id"] = None
    elif anomaly_type == "extreme_amount":
        event["amount"] = round(random.uniform(50000, 999999), 2)
    elif anomaly_type == "invalid_currency":
        event["currency"] = "XYZ"
    elif anomaly_type == "future_timestamp":
        event["timestamp"] = "2099-01-01T00:00:00+00:00"
    elif anomaly_type == "negative_amount":
        event["amount"] = round(random.uniform(-500, -1), 2)
    elif anomaly_type == "schema_drift":
        event["unexpected_field"] = "drift_detected"
        del event["payment_method"]

    event["_anomaly_type"] = anomaly_type
    return event


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )
    logger.info(f"Producer started → topic: {TOPIC}")

    try:
        while True:
            event = (
                anomalous_event()
                if random.random() < ANOMALY_RATE
                else normal_event()
            )
            producer.send(TOPIC, key=event["event_id"].encode(), value=event)
            logger.info(f"Sent: {event['event_id']} | amount={event.get('amount')} | anomaly={event.get('_anomaly_type', 'none')}")
            time.sleep(random.uniform(0.1, 0.5))
    except KeyboardInterrupt:
        logger.info("Producer stopped.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
