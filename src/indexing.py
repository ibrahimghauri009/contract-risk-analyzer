"""Vector Store and Sparse Search Indexing Pipeline.

Combines:
- ChromaDB for Dense Vector Embeddings (BGE-small / MiniLM)
- BM25 for Sparse Keyword Matching
- Per-contract isolation and active session management
"""
import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.config import settings
from src.ingest import ContractChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text for BM25 with simple lowercasing and punctuation stripping."""
    return re.findall(r"\b\w+\b", text.lower())

class HybridIndexManager:
    """Manages both ChromaDB dense vector collection and BM25 sparse index."""

    def __init__(
        self,
        persist_dir: Path = settings.CHROMA_PERSIST_DIR,
        collection_name: str = "contract_clauses",
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)

        # Global Sparse BM25 state
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_store: Dict[str, ContractChunk] = {}
        self.bm25_chunk_ids: List[str] = []
        
        # Active contract isolated cache
        self.active_contract_id: Optional[str] = None
        self.active_chunks: List[ContractChunk] = []
        self.active_bm25: Optional[BM25Okapi] = None

        self._load_bm25_index()

    def set_active_contract(self, chunks: List[ContractChunk], contract_id: str):
        """Builds an isolated in-memory index for the current active contract."""
        self.active_contract_id = contract_id
        self.active_chunks = chunks
        
        # Build contract-specific chunk store
        for c in chunks:
            self.chunk_store[c.chunk_id] = c
            
        corpus_tokens = [tokenize_for_bm25(c.text) for c in chunks]
        if corpus_tokens:
            self.active_bm25 = BM25Okapi(corpus_tokens)

    def index_chunks(self, chunks: List[ContractChunk], contract_id: Optional[str] = None) -> int:
        """Indexes a list of ContractChunk items into ChromaDB and BM25."""
        if not chunks:
            return 0

        target_contract_id = contract_id or chunks[0].contract_id
        self.set_active_contract(chunks, target_contract_id)

        logger.info(f"Indexing {len(chunks)} chunks for contract '{target_contract_id}'...")
        
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "contract_id": c.contract_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "section_number": c.section_number or "",
                "section_title": c.section_title or "",
                "page_number": c.page_number or 1
            }
            for c in chunks
        ]

        # Generate Dense Embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Update global BM25
        for c in chunks:
            if c.chunk_id not in self.bm25_chunk_ids:
                self.bm25_chunk_ids.append(c.chunk_id)

        corpus_tokens = [tokenize_for_bm25(self.chunk_store[cid].text) for cid in self.bm25_chunk_ids if cid in self.chunk_store]
        if corpus_tokens:
            self.bm25 = BM25Okapi(corpus_tokens)
            self._save_bm25_index()

        return len(chunks)

    def _save_bm25_index(self):
        """Persists BM25 corpus and chunk mapping."""
        bm25_file = self.persist_dir / f"{self.collection_name}_bm25.pkl"
        with open(bm25_file, "wb") as f:
            pickle.dump({
                "chunk_ids": self.bm25_chunk_ids,
                "chunk_store": self.chunk_store
            }, f)

    def _load_bm25_index(self):
        """Loads persisted BM25 corpus if available."""
        bm25_file = self.persist_dir / f"{self.collection_name}_bm25.pkl"
        if bm25_file.exists():
            try:
                with open(bm25_file, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_chunk_ids = data["chunk_ids"]
                    self.chunk_store = data["chunk_store"]
                    valid_ids = [cid for cid in self.bm25_chunk_ids if cid in self.chunk_store]
                    corpus_tokens = [tokenize_for_bm25(self.chunk_store[cid].text) for cid in valid_ids]
                    if corpus_tokens:
                        self.bm25 = BM25Okapi(corpus_tokens)
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
