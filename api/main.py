"""FastAPI REST Service for Contract Risk Analyzer.

Endpoints:
- POST /upload: Upload & index PDF/TXT contract
- POST /analyze: Comprehensive risk assessment & clause classification
- POST /query: Hybrid retrieval + Cross-Encoder rerank + citation grounding
- GET /health: Service health & model status
"""
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.ingest import ContractIngestion
from src.indexing import HybridIndexManager
from src.retriever import HybridRetriever
from src.classifier import ClauseClassifier
from src.risk_engine import RiskEngine
from src.generator import GroundedGenerator
from api.schemas import (
    AnalyzeTextRequest,
    RiskReportResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="⚖️ Contract Risk Analyzer API",
    description="Legal-tech RAG + Supervised ML Clause Classifier with Exact Span Grounding",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Singleton Instances
ingestion = ContractIngestion()
index_manager = HybridIndexManager()
retriever = HybridRetriever(index_manager=index_manager)
classifier = ClauseClassifier()
risk_engine = RiskEngine(classifier=classifier)
generator = GroundedGenerator(classifier=classifier)

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "indexed_chunks_count": index_manager.collection.count(),
        "classifier_classes": len(classifier.label_encoder.classes_) if classifier.label_encoder else 0
    }

@app.post("/upload", response_model=UploadResponse, tags=["Contract Ingestion"])
async def upload_contract(file: UploadFile = File(...)):
    """Uploads a PDF or TXT contract, segments it into clause chunks, and indexes it into ChromaDB + BM25."""
    try:
        contract_id = f"contract_{uuid.uuid4().hex[:8]}"
        temp_dir = settings.DATA_DIR / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / f"{contract_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse & Chunk
        full_text, page_maps = ingestion.load_document(file_path)
        chunks = ingestion.chunk_contract(full_text, contract_id=contract_id, page_maps=page_maps)

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract readable text chunks from uploaded document.")

        # Index into ChromaDB + BM25
        num_indexed = index_manager.index_chunks(chunks)

        return UploadResponse(
            contract_id=contract_id,
            filename=file.filename,
            chunks_indexed=num_indexed,
            message=f"Successfully indexed {num_indexed} clause chunks."
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=RiskReportResponse, tags=["Risk Analysis"])
async def analyze_contract_text(request: AnalyzeTextRequest):
    """Analyzes raw contract text, identifies high-risk clauses, detects missing protections, and scores risk (0-100)."""
    try:
        chunks = ingestion.chunk_contract(request.contract_text, contract_id=request.contract_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="Provided text could not be segmented into clause chunks.")

        report = risk_engine.analyze_contract(chunks, contract_id=request.contract_id)
        return report.to_dict()
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Q&A"])
async def query_contract(request: QueryRequest):
    """Performs hybrid retrieval + Cross-Encoder reranking and generates citation-grounded response."""
    try:
        retrieved_clauses = retriever.retrieve(
            query=request.query,
            contract_id=request.contract_id,
            top_k=request.top_k,
            use_reranker=request.use_reranker
        )

        grounded_answer = generator.generate(
            query=request.query,
            retrieved_clauses=retrieved_clauses
        )

        return grounded_answer.to_dict()
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
