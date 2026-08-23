"""Advanced Hybrid Retrieval Pipeline with Cross-Encoder Reranking.

Implements:
1. Dense Semantic Search (ChromaDB)
2. Sparse Keyword Search (BM25)
3. Reciprocal Rank Fusion (RRF)
4. Cross-Encoder Reranker for state-of-the-art precision
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import CrossEncoder
from src.config import settings
from src.ingest import ContractChunk
from src.indexing import HybridIndexManager, tokenize_for_bm25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievedClause:
    chunk_id: str
    contract_id: str
    text: str
    start_char: int
    end_char: int
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class HybridRetriever:
    """Combines Dense, BM25, and Cross-Encoder Reranking for high-precision legal retrieval."""

    def __init__(
        self,
        index_manager: Optional[HybridIndexManager] = None,
        reranker_model_name: str = settings.RERANKER_MODEL_NAME
    ):
        self.index_manager = index_manager or HybridIndexManager()
        self.reranker_model_name = reranker_model_name
        logger.info(f"Loading Cross-Encoder reranker: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name)

    def retrieve(
        self,
        query: str,
        contract_id: Optional[str] = None,
        top_k: int = 5,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        use_reranker: bool = True
    ) -> List[RetrievedClause]:
        """
        Executes hybrid retrieval:
        1. Query dense index (ChromaDB)
        2. Query sparse index (BM25)
        3. Fuse using Reciprocal Rank Fusion (RRF)
        4. Rerank top candidates with Cross-Encoder
        """
        # --- 1. Dense Retrieval ---
        query_embedding = self.index_manager.embedder.encode([query], normalize_embeddings=True).tolist()
        
        where_clause = {"contract_id": contract_id} if contract_id else None
        
        dense_results = self.index_manager.collection.query(
            query_embeddings=query_embedding,
            n_results=min(candidate_pool_size, max(1, self.index_manager.collection.count())),
            where=where_clause
        )

        dense_ranked_ids = []
        if dense_results and "ids" in dense_results and dense_results["ids"]:
            dense_ranked_ids = dense_results["ids"][0]

        # --- 2. Sparse (BM25) Retrieval ---
        sparse_ranked_ids = []
        if self.index_manager.bm25 and self.index_manager.bm25_chunk_ids:
            query_tokens = tokenize_for_bm25(query)
            bm25_scores = self.index_manager.bm25.get_scores(query_tokens)
            
            # Filter by contract_id if specified
            valid_indices = []
            for idx, cid in enumerate(self.index_manager.bm25_chunk_ids):
                chunk = self.index_manager.chunk_store.get(cid)
                if not contract_id or (chunk and chunk.contract_id == contract_id):
                    valid_indices.append(idx)

            if valid_indices:
                sorted_valid = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)
                sparse_ranked_ids = [self.index_manager.bm25_chunk_ids[i] for i in sorted_valid[:candidate_pool_size]]

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for rank, cid in enumerate(dense_ranked_ids):
            dense_ranks[cid] = rank + 1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight * (1.0 / (rrf_k + rank + 1))

        for rank, cid in enumerate(sparse_ranked_ids):
            sparse_ranks[cid] = rank + 1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_weight * (1.0 / (rrf_k + rank + 1))

        if not rrf_scores:
            return []

        # Sort candidate pool by RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool_size]
        
        candidates: List[RetrievedClause] = []
        for cid, score in sorted_candidates:
            chunk = self.index_manager.chunk_store.get(cid)
            if chunk:
                candidates.append(RetrievedClause(
                    chunk_id=chunk.chunk_id,
                    contract_id=chunk.contract_id,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    score=score,
                    dense_rank=dense_ranks.get(cid),
                    sparse_rank=sparse_ranks.get(cid),
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number
                ))

        # --- 4. Cross-Encoder Reranking ---
        if use_reranker and candidates:
            pairs = [[query, c.text] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for c, r_score in zip(candidates, rerank_scores):
                c.rerank_score = float(r_score)
                
            # Sort by reranker score descending
            candidates = sorted(candidates, key=lambda x: x.rerank_score or -999.0, reverse=True)

        return candidates[:top_k]
