# Contract Risk Analyzer

Clause retrieval and risk flagging for commercial contracts. Upload a contract (PDF or text),
ask questions about it, and get answers that quote the exact clause text with character offsets,
plus a rule-based risk report built on a clause-type classifier trained on
[CUAD](https://www.atticusprojectai.org/cuad).

[![CI](https://github.com/ibrahimghauri009/contract-risk-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ibrahimghauri009/contract-risk-analyzer/actions/workflows/ci.yml)

## What it does

- **Structure-aware chunking** splits contracts by section and clause and records exact start/end
  character offsets for every chunk. A test checks that `text[start:end] == chunk.text`.
- **Hybrid retrieval**: dense similarity (`BAAI/bge-small-en-v1.5`) and BM25, combined with weighted
  Reciprocal Rank Fusion, then optionally reranked with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Search is restricted to the active contract.
- **Clause-type classifier**: logistic regression on sentence embeddings, trained on 15 risk-relevant
  CUAD categories (up to 120 clauses each, stratified 80/20 split).
- **Risk engine**: combines classifier predictions with pattern rules (e.g. uncapped liability,
  non-compete duration) and flags missing protective clauses (liability cap, governing law,
  anti-assignment), producing an overall score with per-finding citations.
- **Answers**: extractive by default (the best-matching sentence plus extracted durations, amounts and
  jurisdictions). If an OpenAI API key is set, an LLM writes the answer from the top three retrieved clauses.
- **Interfaces**: FastAPI (`/upload`, `/analyze`, `/query`, `/health`) and a Streamlit dashboard.

## Architecture

```
contract (.pdf/.txt)
   └─ chunker (sections → clauses, exact offsets)
        ├─ dense scores (bge-small) ┐
        └─ BM25 scores              ┴─ weighted RRF ─ optional cross-encoder rerank
                                                         │
                     clause classifier (LogReg on embeddings, 15 CUAD types)
                                                         │
                     risk rules + missing-clause checks ─┴─ answer + citations
                                                         │
                                         FastAPI  /  Streamlit
```

## Evaluation

`src/eval.py` compares three retrieval configurations (dense only, dense + BM25, and dense + BM25 with
cross-encoder reranking) using Recall@1/3/5 and MRR. It matches 15 natural-language questions to CUAD
clause categories and checks whether the gold clause appears in the top results for each contract.
`src/classifier.py` reports per-class precision, recall and F1 on a stratified 20% hold-out of the
clause data.

Figures published in an earlier version of this README were removed: they were not backed by committed
output files, and the "citation accuracy" metric only checked that a result was non-empty. Results will
be added here together with the metric files that produce them.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/ibrahimghauri009/contract-risk-analyzer.git
cd contract-risk-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: add OPENAI_API_KEY for LLM answers
```

Prepare data and the classifier (downloads CUAD from Zenodo on first run):

```bash
python -m src.data_loader        # writes data/processed/cuad_clauses.csv and cuad_contracts.csv
python -m src.classifier         # trains and saves models/classifier.joblib
python -m src.eval               # retrieval benchmark → data/processed/eval_benchmark_results.json
```

## Usage

```bash
uvicorn api.main:app --port 8000          # API docs at http://localhost:8000/docs
streamlit run app/streamlit_app.py        # dashboard
```

Two sample contracts are in `data/test_contracts/`.

Docker (API only):

```bash
docker build -t contract-risk-analyzer .
docker run -p 8000:8000 contract-risk-analyzer
```

## Tests

```bash
pytest -v
```

Chunking, offset and retrieval tests run anywhere. Tests that need the trained classifier
(`python -m src.classifier`) or the processed CUAD files (`python -m src.data_loader`) are skipped with a
reason when those artifacts are missing, which is the case in CI.

## Design decisions

- **Exact offsets over paraphrase.** Every citation is a verbatim slice of the source text, so a reader
  can check it.
- **Rank fusion instead of score mixing.** BM25 and cosine scores sit on different scales, and RRF
  only uses ranks.
- **A small supervised classifier instead of LLM labelling.** It is deterministic, cheap and measurable;
  the LLM is optional and only phrases answers.
- **Extractive fallback.** The system works with no API key.

## Limitations

- The risk score comes from hand-written rules plus classifier output. It has not been validated
  against lawyer judgments and is not legal advice.
- The classifier covers 15 of CUAD's 41 categories and sees only the first 512 characters of each clause.
- Dense and BM25 scoring are recomputed in memory for every query. ChromaDB is populated but not
  yet used for search.
- The LLM output is parsed from free text without validation.
- No authentication; CORS is open. Not intended for deployment as-is.

## Licence

MIT; see `LICENSE`. CUAD is licensed CC BY 4.0 by The Atticus Project.
