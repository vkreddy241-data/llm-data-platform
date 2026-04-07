"""
Unit tests for LLM Data Platform.
Tests pipeline logic, prompt building, and agent tools in isolation
— no real LLM calls, no cloud credentials needed.
Run: pytest tests/ -v
"""

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestQualityFlags:
    """Test Spark-free logic for anomaly flag rules."""

    def _check_flags(self, record: dict) -> dict:
        VALID_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
        MAX_AMOUNT = 10000.0
        return {
            "flag_null_user":          record.get("user_id") is None,
            "flag_extreme_amount":     (record.get("amount") or 0) > MAX_AMOUNT,
            "flag_negative_amount":    (record.get("amount") or 0) < 0,
            "flag_invalid_currency":   record.get("currency") not in VALID_CURRENCIES,
        }

    def test_normal_record_no_flags(self):
        record = {"user_id": "USR-1", "amount": 99.0, "currency": "USD"}
        flags = self._check_flags(record)
        assert not any(flags.values())

    def test_null_user_flagged(self):
        record = {"user_id": None, "amount": 99.0, "currency": "USD"}
        assert self._check_flags(record)["flag_null_user"] is True

    def test_extreme_amount_flagged(self):
        record = {"user_id": "USR-1", "amount": 50000.0, "currency": "USD"}
        assert self._check_flags(record)["flag_extreme_amount"] is True

    def test_negative_amount_flagged(self):
        record = {"user_id": "USR-1", "amount": -10.0, "currency": "USD"}
        assert self._check_flags(record)["flag_negative_amount"] is True

    def test_invalid_currency_flagged(self):
        record = {"user_id": "USR-1", "amount": 10.0, "currency": "XYZ"}
        assert self._check_flags(record)["flag_invalid_currency"] is True

    def test_anomaly_score_accumulates(self):
        record = {"user_id": None, "amount": -5.0, "currency": "XYZ"}
        flags = self._check_flags(record)
        score = sum(int(v) for v in flags.values())
        assert score == 3


# ---------------------------------------------------------------------------
# LLM prompt tests
# ---------------------------------------------------------------------------

class TestAnomalyPrompt:
    def test_prompt_contains_anomaly_data(self):
        from ai.anomaly_detector import build_analysis_prompt
        anomalies = [
            {"event_id": "EVT-001", "_anomaly_type": "null_user", "amount": 100.0},
            {"event_id": "EVT-002", "_anomaly_type": "extreme_amount", "amount": 99999.0},
        ]
        prompt = build_analysis_prompt(anomalies)
        assert "EVT-001" in prompt
        assert "null_user" in prompt
        assert "extreme_amount" in prompt
        assert "ROOT CAUSE" in prompt
        assert "BUSINESS IMPACT" in prompt

    def test_prompt_includes_record_count(self):
        from ai.anomaly_detector import build_analysis_prompt
        anomalies = [{"event_id": f"EVT-{i}"} for i in range(5)]
        prompt = build_analysis_prompt(anomalies)
        assert "5" in prompt


class TestSchemaPrompt:
    def test_schema_prompt_includes_drift(self):
        from ai.schema_analyzer import build_schema_prompt
        drift = {
            "from_version": 3,
            "to_version": 4,
            "added_columns": ["new_field"],
            "removed_columns": ["old_field"],
            "type_changes": [],
        }
        history = [
            {"version": 4, "timestamp": "2024-06-02", "columns": [{"name": "new_field", "type": "StringType", "nullable": True}]},
            {"version": 3, "timestamp": "2024-06-01", "columns": [{"name": "old_field", "type": "StringType", "nullable": True}]},
        ]
        prompt = build_schema_prompt(drift, history)
        assert "new_field" in prompt
        assert "old_field" in prompt
        assert "BREAKING CHANGE" in prompt


class TestEnrichmentPrompt:
    def test_enrichment_prompt_structure(self):
        from ai.data_enricher import build_enrichment_prompt
        records = [{"event_id": "EVT-001", "amount": 299.0, "country": "US"}]
        prompt = build_enrichment_prompt(records)
        assert "fraud_risk" in prompt
        assert "customer_intent" in prompt
        assert "enrichment_confidence" in prompt
        assert "EVT-001" in prompt


# ---------------------------------------------------------------------------
# RAG Catalog tests (no LLM calls)
# ---------------------------------------------------------------------------

class TestRAGCatalog:
    def test_rag_prompt_includes_question(self):
        from catalog.rag_catalog import RAGDataCatalog
        context_tables = [
            {
                "document": "TABLE: transactions_clean\nContains clean transaction data.",
                "metadata": {"table_name": "transactions_clean", "delta_path": "s3a://test/clean"},
            }
        ]
        prompt = RAGDataCatalog.build_rag_prompt("Which table has amounts?", context_tables)
        assert "Which table has amounts?" in prompt
        assert "transactions_clean" in prompt

    def test_no_tables_returns_helpful_message(self):
        from catalog.rag_catalog import RAGDataCatalog
        catalog = RAGDataCatalog.__new__(RAGDataCatalog)
        catalog.client = MagicMock()
        catalog.collection = MagicMock()
        catalog.collection.count.return_value = 0
        catalog.collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        result = catalog.query("Which table has amounts?")
        assert "No tables" in result["answer"]
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# API tests (no real server needed)
# ---------------------------------------------------------------------------

class TestAPIModels:
    def test_catalog_query_model(self):
        from api.main import CatalogQuery
        q = CatalogQuery(question="Which table has fraud data?")
        assert q.question == "Which table has fraud data?"
        assert q.top_k == 5

    def test_enrich_request_model(self):
        from api.main import EnrichRequest
        r = EnrichRequest(
            event_id="EVT-001", user_id="USR-1", product_id="PRD-1",
            amount=99.9, currency="USD", event_type="purchase",
            country="US", payment_method="card",
        )
        assert r.amount == 99.9
        assert r.currency == "USD"

    def test_health_response(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pipeline" in data

    def test_root_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "LLM Data Platform" in response.json()["service"]
