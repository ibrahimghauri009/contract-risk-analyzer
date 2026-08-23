"""CUAD Dataset Loader and Parser.

Downloads the CUAD dataset (SQuAD format JSON / CSV),
extracts labeled clause spans across the 41 CUAD legal categories,
formats positive and negative examples for classifier training,
and prepares contract texts for RAG indexing.
"""
import os
import json
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from tqdm import tqdm
import pandas as pd
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CUAD_ZENODO_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
CUAD_HF_JSON_URL = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"

class CUADDataLoader:
    def __init__(
        self,
        raw_dir: Path = settings.RAW_DATA_DIR,
        processed_dir: Path = settings.PROCESSED_DATA_DIR
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.raw_dir / "CUAD_v1.json"

    def download_dataset(self) -> Path:
        """Downloads CUAD_v1.json if not already present."""
        if self.json_path.exists() and self.json_path.stat().st_size > 1000000:
            logger.info(f"CUAD dataset already exists at {self.json_path}")
            return self.json_path

        logger.info(f"Downloading CUAD dataset from {CUAD_HF_JSON_URL}...")
        try:
            response = requests.get(CUAD_HF_JSON_URL, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            
            with open(self.json_path, "wb") as f, tqdm(
                desc="Downloading CUAD_v1.json",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    size = f.write(chunk)
                    bar.update(size)
            logger.info(f"Downloaded CUAD_v1.json successfully ({self.json_path.stat().st_size / (1024*1024):.1f} MB).")
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}. Trying Zenodo archive...")
            self._download_from_zenodo()

        return self.json_path

    def _download_from_zenodo(self):
        """Fallback to downloading CUAD zip from Zenodo."""
        zip_path = self.raw_dir / "CUAD_v1.zip"
        response = requests.get(CUAD_ZENODO_URL, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading CUAD_v1.zip",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                size = f.write(chunk)
                bar.update(size)
                
        logger.info("Extracting CUAD_v1.json from zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith("CUAD_v1.json"):
                    with zip_ref.open(member) as source, open(self.json_path, "wb") as target:
                        target.write(source.read())
                    break

    def extract_dataset(
        self,
        max_contracts: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses CUAD_v1.json into:
        1. `clauses_df`: Labeled positive clause spans (category, text, offsets, contract)
        2. `contracts_df`: Full contract texts and metadata for RAG indexing
        """
        self.download_dataset()
        
        logger.info(f"Parsing {self.json_path}...")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data_list = data.get("data", [])
        logger.info(f"Found {len(data_list)} total contracts in dataset.")

        clause_records = []
        contract_records = []

        for idx, item in enumerate(tqdm(data_list, desc="Extracting contracts & clauses")):
            if max_contracts and idx >= max_contracts:
                break

            title = item.get("title", f"Contract_{idx}")
            paragraphs = item.get("paragraphs", [])

            for p_idx, para in enumerate(paragraphs):
                context = para.get("context", "")
                contract_records.append({
                    "contract_id": title,
                    "paragraph_id": p_idx,
                    "text": context,
                    "char_count": len(context),
                    "word_count": len(context.split())
                })

                qas = para.get("qas", [])
                for qa in qas:
                    question = qa.get("question", "")
                    category = self._clean_category_name(question)
                    answers = qa.get("answers", [])
                    is_impossible = qa.get("is_impossible", False)

                    if not is_impossible and answers:
                        for ans in answers:
                            ans_text = ans.get("text", "").strip()
                            ans_start = ans.get("answer_start", 0)
                            if ans_text:
                                clause_records.append({
                                    "contract_id": title,
                                    "category": category,
                                    "clause_text": ans_text,
                                    "start_offset": int(ans_start),
                                    "end_offset": int(ans_start + len(ans_text)),
                                    "is_clause": 1
                                })

        clauses_df = pd.DataFrame(clause_records)
        contracts_df = pd.DataFrame(contract_records)

        logger.info(f"Extracted {len(clauses_df)} labeled clause instances across {clauses_df['category'].nunique()} categories.")
        logger.info(f"Extracted {len(contracts_df)} contract paragraphs.")

        # Save to processed directory
        clauses_df.to_csv(self.processed_dir / "cuad_clauses.csv", index=False)
        contracts_df.to_csv(self.processed_dir / "cuad_contracts.csv", index=False)
        
        return clauses_df, contracts_df

    def _clean_category_name(self, question: str) -> str:
        """Extracts clean clause category name from CUAD question prompt."""
        if '"' in question:
            return question.split('"')[1].strip()
        return question.strip()

if __name__ == "__main__":
    loader = CUADDataLoader()
    clauses_df, contracts_df = loader.extract_dataset()
    print("\n--- Clause Categories Distribution (Top 15) ---")
    print(clauses_df["category"].value_counts().head(15))
    print(f"\nTotal clauses: {len(clauses_df)}")
    print(f"Total contracts: {len(contracts_df)}")
    print(f"Saved datasets to {settings.PROCESSED_DATA_DIR}")
