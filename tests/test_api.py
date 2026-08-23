"""Test suite for FastAPI REST API endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_TEXT = """
AGREEMENT
1. GOVERNING LAW: Governed by the laws of New York.
2. LIMITATION OF LIABILITY: Total aggregate liability is strictly capped at $10,000.
3. NON-COMPETE: Provider agrees not to compete for 3 years.
"""

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data

def test_analyze_endpoint():
    payload = {
        "contract_text": SAMPLE_TEXT,
        "contract_id": "api_test_contract"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["contract_id"] == "api_test_contract"
    assert "overall_score" in data
    assert "risk_level" in data
    assert len(data["findings"]) > 0

def test_query_endpoint():
    # First analyze so chunks exist in engine if needed, or query directly
    payload = {
        "query": "What is the liability limitation?",
        "top_k": 3,
        "use_reranker": True
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
