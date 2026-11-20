"""Shared pytest configuration.

Some tests need artifacts that are not committed: the trained classifier
(`python -m src.classifier`) and the processed CUAD files (`python -m src.data_loader`).
Those tests are marked and skipped, with a reason, when the artifacts are absent (as in CI).
"""
import pytest
from src.config import settings

HAS_MODEL = settings.CLASSIFIER_PATH.exists() and settings.LABEL_ENCODER_PATH.exists()
HAS_CUAD = (settings.PROCESSED_DATA_DIR / "cuad_clauses.csv").exists() and     (settings.PROCESSED_DATA_DIR / "cuad_contracts.csv").exists()

requires_model = pytest.mark.skipif(
    not HAS_MODEL, reason="trained classifier not found; run `python -m src.classifier`")
requires_cuad = pytest.mark.skipif(
    not HAS_CUAD, reason="processed CUAD data not found; run `python -m src.data_loader`")
