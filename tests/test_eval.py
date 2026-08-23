"""Test suite for offline evaluation benchmark."""
import pytest
from src.eval import BenchmarkEvaluator

def test_evaluator_initialization():
    evaluator = BenchmarkEvaluator()
    assert evaluator.contracts_csv.exists()
    assert evaluator.clauses_csv.exists()
