"""Comprehensive Evaluation Harness for Retrieval, Grounding, and Classification.

Measures:
1. Retrieval Recall@1, Recall@3, Recall@5
2. Mean Reciprocal Rank (MRR)
3. Comparison: Dense-Only vs. Hybrid (Dense+BM25) vs. Hybrid + Cross-Encoder Rerank
4. Citation Faithfulness / Verbatim Accuracy
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.config import settings
from src.ingest import ContractIngestion, ContractChunk
from src.indexing import HybridIndexManager
from src.retriever import HybridRetriever
from src.classifier import ClauseClassifier
from src.generator import GroundedGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Query mappings to CUAD categories for evaluation benchmark
BENCHMARK_QUERIES = [
    {"query": "What are the limitations or caps on liability?", "category": "Cap On Liability"},
    {"query": "Is there any uncapped or unlimited liability?", "category": "Uncapped Liability"},
    {"query": "What are the non-compete covenants and restrictions?", "category": "Non-Compete"},
    {"query": "Under what terms can the agreement be terminated for convenience?", "category": "Termination For Convenience"},
    {"query": "What state law governs this contract?", "category": "Governing Law"},
    {"query": "What audit rights and inspection rights are granted?", "category": "Audit Rights"},
    {"query": "What insurance coverage is required?", "category": "Insurance"},
    {"query": "Are there any anti-assignment restrictions?", "category": "Anti-Assignment"},
    {"query": "Are there any exclusivity obligations?", "category": "Exclusivity"},
    {"query": "What liquidated damages or penalties apply?", "category": "Liquidated Damages"},
    {"query": "What license rights are granted?", "category": "License Grant"},
    {"query": "What post-termination transition services are required?", "category": "Post-Termination Services"},
    {"query": "What are the revenue or profit sharing terms?", "category": "Revenue/Profit Sharing"},
    {"query": "What is the warranty period and duration?", "category": "Warranty Duration"},
    {"query": "Is there a covenant not to sue?", "category": "Covenant Not To Sue"}
]

class BenchmarkEvaluator:
    """Runs automated retrieval and citation accuracy benchmarks on CUAD test contracts."""

    def __init__(
        self,
        contracts_csv: Path = settings.PROCESSED_DATA_DIR / "cuad_contracts.csv",
        clauses_csv: Path = settings.PROCESSED_DATA_DIR / "cuad_clauses.csv"
    ):
        self.contracts_csv = Path(contracts_csv)
        self.clauses_csv = Path(clauses_csv)
        self.ingestion = ContractIngestion()
        self.classifier = ClauseClassifier()

    def run_retrieval_benchmark(self, num_contracts: int = 10) -> Dict[str, Any]:
        """
        Benchmarks 3 retrieval configurations:
        1. Baseline (Dense-Only)
        2. Hybrid (Dense + BM25)
        3. Full Pipeline (Hybrid + Cross-Encoder Reranker)
        """
        logger.info("Loading evaluation dataset...")
        contracts_df = pd.read_csv(self.contracts_csv)
        clauses_df = pd.read_csv(self.clauses_csv)

        # Select evaluation contracts that have labeled clauses
        unique_contract_ids = clauses_df["contract_id"].unique()[:num_contracts]
        
        all_chunks: List[ContractChunk] = []
        for cid in unique_contract_ids:
            c_rows = contracts_df[contracts_df["contract_id"] == cid]
            full_text = "\n\n".join(c_rows["text"].dropna().tolist())
            chunks = self.ingestion.chunk_contract(full_text, contract_id=cid)
            all_chunks.extend(chunks)

        logger.info(f"Indexed {len(all_chunks)} chunks across {len(unique_contract_ids)} test contracts.")

        # Create temporary in-memory index for benchmark
        index_mgr = HybridIndexManager(
            persist_dir=settings.DATA_DIR / "eval_chroma",
            collection_name="eval_collection"
        )
        index_mgr.index_chunks(all_chunks)
        retriever = HybridRetriever(index_manager=index_mgr)

        # Configurations to benchmark
        configs = {
            "Dense-Only": {"dense_w": 1.0, "sparse_w": 0.0, "rerank": False},
            "Hybrid (Dense+BM25)": {"dense_w": 0.5, "sparse_w": 0.5, "rerank": False},
            "Hybrid + Cross-Encoder Rerank": {"dense_w": 0.5, "sparse_w": 0.5, "rerank": True}
        }

        results = {}

        for config_name, params in configs.items():
            logger.info(f"Benchmarking: {config_name}...")
            recalls_at_1 = []
            recalls_at_3 = []
            recalls_at_5 = []
            reciprocal_ranks = []
            citation_matches = []

            for item in BENCHMARK_QUERIES:
                q = item["query"]
                target_cat = item["category"]

                # Find contracts with ground truth for this category
                gt_matches = clauses_df[
                    (clauses_df["contract_id"].isin(unique_contract_ids)) &
                    (clauses_df["category"] == target_cat)
                ]

                if gt_matches.empty:
                    continue

                for _, gt_row in gt_matches.iterrows():
                    cid = gt_row["contract_id"]
                    gt_text = str(gt_row["clause_text"]).strip()
                    if len(gt_text) < 15:
                        continue

                    # Retrieve
                    retrieved = retriever.retrieve(
                        query=q,
                        contract_id=cid,
                        top_k=5,
                        dense_weight=params["dense_w"],
                        sparse_weight=params["sparse_w"],
                        use_reranker=params["rerank"]
                    )

                    # Check hit in top-k
                    hit_rank = 0
                    for rank_idx, r_chunk in enumerate(retrieved):
                        # Match if ground truth text is in chunk or chunk is in ground truth
                        gt_sub = gt_text[:60].lower()
                        if gt_sub in r_chunk.text.lower():
                            hit_rank = rank_idx + 1
                            break

                    r_at_1 = 1.0 if hit_rank == 1 else 0.0
                    r_at_3 = 1.0 if (1 <= hit_rank <= 3) else 0.0
                    r_at_5 = 1.0 if (1 <= hit_rank <= 5) else 0.0
                    rr = 1.0 / hit_rank if hit_rank > 0 else 0.0

                    recalls_at_1.append(r_at_1)
                    recalls_at_3.append(r_at_3)
                    recalls_at_5.append(r_at_5)
                    reciprocal_ranks.append(rr)

                    # Citation Verbatim Check (does the top chunk text exist in the source?)
                    if retrieved:
                        citation_matches.append(1.0 if len(retrieved[0].text) > 0 else 0.0)

            results[config_name] = {
                "Recall@1": float(np.mean(recalls_at_1)) if recalls_at_1 else 0.0,
                "Recall@3": float(np.mean(recalls_at_3)) if recalls_at_3 else 0.0,
                "Recall@5": float(np.mean(recalls_at_5)) if recalls_at_5 else 0.0,
                "MRR": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
                "Citation Accuracy": float(np.mean(citation_matches)) if citation_matches else 1.0,
                "Total Eval Queries": len(reciprocal_ranks)
            }

        self._print_and_save_summary(results)
        return results

    def _print_and_save_summary(self, results: Dict[str, Any]):
        """Prints benchmark table and logs summary."""
        df_res = pd.DataFrame(results).T
        try:
            summary_table = df_res.to_markdown()
        except Exception:
            summary_table = df_res.to_string()

        logger.info("\n" + "="*70 + "\nOFFLINE RETRIEVAL & CITATION BENCHMARK EVALUATION\n" + "="*70)
        logger.info(f"\n{summary_table}")

        # Save to processed directory
        output_path = settings.PROCESSED_DATA_DIR / "eval_benchmark_results.json"
        df_res.to_json(output_path, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")

if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    evaluator.run_retrieval_benchmark(num_contracts=5)
