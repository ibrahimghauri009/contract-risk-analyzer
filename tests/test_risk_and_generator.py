"""Test suite for Risk Engine and Grounded Generator."""
import pytest
from src.ingest import ContractIngestion
from src.classifier import ClauseClassifier
from src.risk_engine import RiskEngine, RiskSeverity
from src.retriever import RetrievedClause
from src.generator import GroundedGenerator

RISKY_CONTRACT_TEXT = """
MASTER CONSULTING AGREEMENT

1. UNLIMITED LIABILITY
NEITHER PARTY LIMITS ITS LIABILITY FOR ANY CLAIMS, LOSSES, OR DAMAGES WHATSOEVER ARISING UNDER THIS AGREEMENT.

2. NON-COMPETE COVENANT
For a period of 5 years following termination of this agreement, Provider shall not engage in any competitive software business worldwide.

3. EXCLUSIVITY
Client shall exclusively procure all software advisory services from Provider and from no other third party.

4. TERMINATION FOR CONVENIENCE
Client may terminate immediately without notice at any time.
"""

def test_risk_engine_evaluation():
    ingestion = ContractIngestion()
    chunks = ingestion.chunk_contract(RISKY_CONTRACT_TEXT, contract_id="risky_test_001")
    
    classifier = ClauseClassifier()
    engine = RiskEngine(classifier=classifier)
    
    report = engine.analyze_contract(chunks, contract_id="risky_test_001")
    
    assert report.overall_score >= 60
    assert report.risk_level in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
    assert len(report.findings) > 0
    
    # Check that critical findings are flagged
    finding_titles = [f.title for f in report.findings]
    assert any("Liability" in t for t in finding_titles)
    assert any("Non-Compete" in t for t in finding_titles)

def test_grounded_generator_citations():
    classifier = ClauseClassifier()
    generator = GroundedGenerator(classifier=classifier)
    
    sample_retrieved = [
        RetrievedClause(
            chunk_id="chunk_01",
            contract_id="test_contract",
            text="This agreement is governed by the laws of Delaware.",
            start_char=100,
            end_char=152,
            score=0.95,
            section_title="Governing Law"
        )
    ]
    
    result = generator.generate("What is the governing law?", sample_retrieved)
    assert len(result.citations) == 1
    assert result.citations[0]["quote"] == "This agreement is governed by the laws of Delaware."
    assert result.citations[0]["start_char"] == 100
    assert result.citations[0]["end_char"] == 152
    assert "Delaware" in result.answer
