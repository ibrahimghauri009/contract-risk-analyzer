"""Pydantic Request and Response Schemas for FastAPI API."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AnalyzeTextRequest(BaseModel):
    contract_text: str = Field(..., description="Raw text of the contract to analyze")
    contract_id: Optional[str] = Field("contract_upload", description="Optional identifier for the contract")

class CitationSchema(BaseModel):
    citation_id: str
    chunk_id: str
    quote: str
    start_char: int
    end_char: int
    page_number: int
    section_title: str

class FindingSchema(BaseModel):
    category: str
    severity: str
    title: str
    description: str
    recommendation: str
    citation_text: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunk_id: Optional[str] = None

class RiskReportResponse(BaseModel):
    contract_id: str
    overall_score: int
    risk_level: str
    total_clauses_analyzed: int
    findings: List[FindingSchema]
    detected_categories: List[str]
    missing_protective_clauses: List[str]

class QueryRequest(BaseModel):
    query: str = Field(..., description="Legal question or clause category to search for")
    contract_id: Optional[str] = Field(None, description="Optional contract ID to filter search")
    top_k: int = Field(5, description="Number of top clauses to retrieve")
    use_reranker: bool = Field(True, description="Whether to apply Cross-Encoder reranking")

class RetrievedClauseSchema(BaseModel):
    chunk_id: str
    contract_id: str
    text: str
    start_char: int
    end_char: int
    score: float
    rerank_score: Optional[float] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    risk_level: str
    risk_notes: str
    citations: List[CitationSchema]
    retrieved_clauses: List[RetrievedClauseSchema]

class UploadResponse(BaseModel):
    contract_id: str
    filename: str
    chunks_indexed: int
    message: str
