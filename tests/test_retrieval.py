"""Test suite for Ingestion, Chunking, Indexing, and Hybrid Retrieval."""
import pytest
from pathlib import Path
from src.ingest import ContractIngestion, ContractChunk
from src.indexing import HybridIndexManager
from src.retriever import HybridRetriever

SAMPLE_CONTRACT_TEXT = """
COMMERCIAL SERVICES AGREEMENT

This Commercial Services Agreement (the "Agreement") is made effective as of January 15, 2024, by and between Alpha Corp ("Client") and Beta LLC ("Provider").

1. SERVICES AND DELIVERABLES
Provider shall deliver software consulting and technical risk advisory services as set forth in Exhibit A.

2. GOVERNING LAW AND JURISDICTION
This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware, without regard to its conflict of laws principles. Any legal action arising out of this Agreement shall be brought exclusively in the state or federal courts located in Wilmington, Delaware.

3. LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES. PROVIDER'S TOTAL AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL BE STRICTLY CAPPED AT THE TOTAL FEES PAID BY CLIENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

4. NON-COMPETE AND NON-SOLICITATION
During the term of this Agreement and for a period of twenty-four (24) months thereafter, Provider shall not directly engage in any business that competes with Client's core risk analysis products within the United States. Furthermore, neither party shall solicit the employees of the other party.

5. TERMINATION FOR CONVENIENCE
Client may terminate this Agreement at any time for convenience upon giving thirty (30) days prior written notice to Provider.
"""

def test_chunking_offsets():
    ingestion = ContractIngestion()
    chunks = ingestion.chunk_contract(
        text=SAMPLE_CONTRACT_TEXT,
        contract_id="sample_test_001"
    )
    
    assert len(chunks) >= 4
    for chunk in chunks:
        # Verify that start and end offsets match exact substring in original text
        extracted_slice = SAMPLE_CONTRACT_TEXT[chunk.start_char:chunk.end_char]
        assert extracted_slice == chunk.text
        assert chunk.contract_id == "sample_test_001"

def test_hybrid_indexing_and_retrieval(tmp_path):
    # Test indexing
    ingestion = ContractIngestion()
    chunks = ingestion.chunk_contract(
        text=SAMPLE_CONTRACT_TEXT,
        contract_id="contract_test_retrieval"
    )

    index_mgr = HybridIndexManager(
        persist_dir=tmp_path / "test_chroma",
        collection_name="test_collection"
    )
    indexed_count = index_mgr.index_chunks(chunks)
    assert indexed_count == len(chunks)

    # Test retrieval
    retriever = HybridRetriever(index_manager=index_mgr)
    
    # Query for Governing Law
    results_law = retriever.retrieve("What state law governs this contract?", top_k=2)
    assert len(results_law) > 0
    assert "Delaware" in results_law[0].text

    # Query for Liability Cap
    results_liab = retriever.retrieve("What is the aggregate liability cap?", top_k=2)
    assert len(results_liab) > 0
    assert "CAPPED" in results_liab[0].text or "LIABILITY" in results_liab[0].text

    # Query for Non-Compete duration
    results_nc = retriever.retrieve("How long is the non-compete restriction?", top_k=2)
    assert len(results_nc) > 0
    assert "twenty-four (24) months" in results_nc[0].text
