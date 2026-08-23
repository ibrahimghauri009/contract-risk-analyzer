"""Test suite for Supervised Clause Classifier."""
import pytest
from src.classifier import ClauseClassifier

@pytest.fixture(scope="module")
def classifier():
    return ClauseClassifier()

def test_classifier_predictions(classifier):
    # Test Governing Law
    sample_law = "This Agreement and all acts and transactions acting hereunder shall be governed by Delaware law."
    pred_law = classifier.predict(sample_law)
    assert pred_law["predicted_category"] == "Governing Law"
    assert pred_law["is_confident"] is True

    # Test Non-Compete
    sample_nc = "During the term and for 2 years thereafter, the employee shall not engage in any competitive business."
    pred_nc = classifier.predict(sample_nc)
    assert pred_nc["predicted_category"] == "Non-Compete"
    assert pred_nc["is_confident"] is True

    # Test Termination
    sample_term = "Either party may terminate this agreement at any time for convenience upon 30 days written notice."
    pred_term = classifier.predict(sample_term)
    assert pred_term["predicted_category"] == "Termination For Convenience"
    assert pred_term["is_confident"] is True

    # Test Liability Cap
    sample_liab = "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INCIDENTAL, SPECIAL OR CONSEQUENTIAL DAMAGES. LIABILITY SHALL BE CAPPED AT THE FEES PAID."
    pred_liab = classifier.predict(sample_liab)
    assert pred_liab["predicted_category"] in ["Cap On Liability", "Uncapped Liability"]
    assert pred_liab["is_confident"] is True

    # Test Anti-Assignment
    sample_assign = "Neither party may assign or transfer its rights or obligations without prior written consent."
    pred_assign = classifier.predict(sample_assign)
    assert pred_assign["predicted_category"] == "Anti-Assignment"
    assert pred_assign["is_confident"] is True
