"""Test suite for offline evaluation benchmark."""
import pytest
from src.eval import BenchmarkEvaluator
from tests.conftest import requires_cuad

pytestmark = requires_cuad

def test_evaluator_initialization():
    evaluator = BenchmarkEvaluator()
    assert evaluator.contracts_csv.exists()
    assert evaluator.clauses_csv.exists()
