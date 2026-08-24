# ⚖️ Contract Risk Analyzer

> **Legal-Tech RAG + Supervised ML Clause Classifier with Exact Span Grounding & Rigorous Evaluation**

[![CI Test Suite](https://github.com/ibrahimghauri009/contract-risk-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ibrahimghauri009/contract-risk-analyzer/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00.svg)](https://www.trychroma.com)
[![HuggingFace](https://img.shields.io/badge/Sentence--Transformers-Embeddings-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade legal AI pipeline designed to ingest commercial contracts, classify critical clause types (using supervised models trained on Stanford's **CUAD** benchmark), detect high-risk liabilities, and retrieve precise, verbatim-grounded answers with exact span citations.

---

## 🎯 Key Highlights

- **Dual-Engine Architecture**: Blends classical discriminative ML (XGBoost / Sentence-Transformers) for fast, deterministic clause classification with Generative RAG for natural-language risk synthesis.
- **Advanced Retrieval**: Hybrid Search (Dense `bge-small-en-v1.5` + Sparse `BM25`) coupled with a Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`).
- **Verbatim Span Grounding**: Zero-hallucination policy requiring all answers to cite verifiable clause spans and character offsets.
- **Scientific Evaluation Suite**: Offline evaluation measuring **Retrieval Recall@k**, **MRR**, and **Citation Faithfulness** on held-out test contracts.
- **Production-Ready**: Containerized with Docker, REST API with FastAPI, and an interactive Streamlit risk dashboard.

---

## 🏗️ Architecture

```
[Contract Upload (.pdf / .txt)]
            │
            ▼
[Legal-Aware Section & Clause Chunker]
            │
            ├───► [Dense Embeddings (BGE-small)] ───┐
            │                                      ├───► [Hybrid Search & Fusion]
            └───► [Sparse Index (BM25)] ───────────┘              │
                                                                   ▼
                                                       [Cross-Encoder Reranker]
                                                                   │
                                                                   ├───► [Supervised Clause Classifier]
                                                                   │     (XGBoost / CUAD 41 Categories)
                                                                   │
                                                                   ▼
                                                       [LLM Grounded Risk Engine]
                                                       (Grounded Synthesis + Risk Rules)
                                                                   │
                                                                   ▼
                                                       [FastAPI & UI Dashboard]
                                                       - Overall Risk Score & Breakdown
                                                       - Flagged Dangerous Clauses
                                                       - Verbatim Citation Links
```

---

## 📊 Benchmark & Evaluation (CUAD Test Split)

| Component | Metric | Baseline (Dense-Only) | Hybrid (Dense + BM25) | **Contract Risk Analyzer (Hybrid + Cross-Encoder)** |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval** | Recall@3 | 44.19% | 51.16% | **51.16% (+15.8% rel.)** |
| **Retrieval** | Recall@5 | 65.12% | 60.47% | **67.44%** |
| **Retrieval** | MRR (Mean Reciprocal Rank) | 0.4240 | 0.4097 | **0.4403** |
| **Classifier** | Macro F1 (15 Key Risk Types) | *N/A* | *N/A* | **81.11%** |
| **Classifier** | Overall Accuracy | *N/A* | *N/A* | **81.11%** |
| **Generation** | Citation Grounding Accuracy | *N/A* | *N/A* | **100.0%** (Exact Span Verbatim) |

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.11
- Git

### 2. Clone & Setup Environment
```bash
git clone https://github.com/ibrahimghauri009/contract-risk-analyzer.git
cd contract-risk-analyzer

# Create virtual environment
py -3.11 -m venv .venv
# Activate virtual environment (Windows)
.venv\Scripts\activate
# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

---

## 📂 Project Structure

```text
contract-risk-analyzer/
├── data/                    # Datasets (raw CUAD, processed splits, sample contracts)
├── models/                  # Trained classifier models and label encoders
├── src/                     # Core business logic (ingest, indexing, classifier, risk engine, eval)
├── api/                     # FastAPI REST API endpoints & Pydantic schemas
├── app/                     # Streamlit web dashboard
├── tests/                   # Automated unit & integration tests
├── Dockerfile               # Production container image
├── requirements.txt         # Pinned project dependencies
└── README.md
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
