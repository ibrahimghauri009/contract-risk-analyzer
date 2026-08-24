"""Advanced Hybrid Retrieval Pipeline with Cross-Encoder Reranking.

Implements:
1. Dense Semantic Search (ChromaDB)
2. Sparse Keyword Search (BM25)
3. Active Contract Isolation & Reciprocal Rank Fusion (RRF)
4. Cross-Encoder Reranker for pinpoint precision
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from rank_bm25 import BM25Okapi
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
        1. Query dense index (ChromaDB or active contract chunks)
        2. Query sparse index (BM25)
        3. Fuse using Reciprocal Rank Fusion (RRF)
        4. Rerank top candidates with Cross-Encoder
        """
        target_contract_id = contract_id or self.index_manager.active_contract_id
        
        # If we have an active contract loaded with chunks, we can search directly within it
        active_chunks = [
            c for c in (self.index_manager.active_chunks or self.index_manager.chunk_store.values())
            if not target_contract_id or c.contract_id == target_contract_id
        ]

        if not active_chunks and self.index_manager.chunk_store:
            active_chunks = list(self.index_manager.chunk_store.values())

        if not active_chunks:
            return []

        # --- 1. Dense Semantic Scoring ---
        query_embedding = self.index_manager.embedder.encode([query], normalize_embeddings=True)
        chunk_texts = [c.text for c in active_chunks]
        chunk_embeddings = self.index_manager.embedder.encode(chunk_texts, normalize_embeddings=True)
        
        # Cosine similarities (since embeddings are normalized, dot product = cosine sim)
        dense_sims = np.dot(chunk_embeddings, query_embedding.T).flatten()
        dense_sorted_indices = np.argsort(dense_sims)[::-1]
        dense_ranked_ids = [active_chunks[i].chunk_id for i in dense_sorted_indices[:candidate_pool_size]]

        # --- 2. Sparse (BM25) Scoring ---
        query_tokens = tokenize_for_bm25(query)
        corpus_tokens = [tokenize_for_bm25(c.text) for c in active_chunks]
        
        sparse_ranked_ids = []
        if corpus_tokens:
            contract_bm25 = BM25Okapi(corpus_tokens)
            bm25_scores = contract_bm25.get_scores(query_tokens)
            sparse_sorted_indices = np.argsort(bm25_scores)[::-1]
            sparse_ranked_ids = [active_chunks[i].chunk_id for i in sparse_sorted_indices[:candidate_pool_size]]

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

        # Sort candidate pool by RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool_size]
        
        candidates: List[RetrievedClause] = []
        for cid, score in sorted_candidates:
            chunk = self.index_manager.chunk_store.get(cid)
            if not chunk:
                # search in active_chunks
                for ac in active_chunks:
                    if ac.chunk_id == cid:
                        chunk = ac
                        break

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
            candidates = sorted(candidates, key=lambda x: x.rerank_score if x.rerank_score is not None else -999.0, reverse=True)

        return candidates[:top_k]
