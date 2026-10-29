"""
tests/test_explainability.py
==============================
Unit tests for the hallucination detector and the ECS metric.
"""
from evaluation.ecs import ExplainabilityConfidenceScore
from llm.hallucination_detector import HallucinationDetector
from backend.service import _plain_language_summary
import numpy as np


def test_hallucination_detector_flags_ungrounded_capitalized_claims():
    detector = HallucinationDetector()
    evidence_features = [{"feature": "price_scaled", "shap_value": 0.5, "direction": "increases_rank"}]
    grounded_text = "The price scaled feature had the strongest measured influence."
    ungrounded_text = "This product was endorsed by CelebrityXYZ and costs $9999999."

    grounded_result = detector.verify(grounded_text, evidence_features, [])
    ungrounded_result = detector.verify(ungrounded_text, evidence_features, [])

    assert grounded_result.confidence_score >= ungrounded_result.confidence_score


def test_ecs_score_is_bounded_between_0_and_1():
    ecs = ExplainabilityConfidenceScore()
    result = ecs.compute(
        all_shap_values=np.array([0.5, 0.2, -0.1, 0.05]),
        top_k_indices=[0, 1],
        graph_path=[{"from": "User:U1", "to": "Product:P1", "relation": "PURCHASED"}],
        hallucination_confidence=0.8,
        features_run_1=["price_scaled", "category_encoded"],
    )
    assert 0.0 <= result.score <= 1.0


def test_plain_language_summary_uses_readable_grounded_feature_names():
    summary = _plain_language_summary(
        [
            {"feature": "price_scaled", "shap_value": 0.4},
            {"feature": "num_reviews_scaled", "shap_value": 0.2},
            {"feature": "graph_degree", "shap_value": -0.1},
        ],
        "Example phone",
    )

    assert "Example phone" in summary
    assert "listed price" in summary
    assert "customer ratings and reviews" in summary
    assert "signals that pulled the score down included connections to other catalog items" in summary.lower()
    assert "SHAP" not in summary


def test_plain_language_comparison_names_which_product_each_signal_favors():
    summary = _plain_language_summary(
        [
            {"feature": "price_scaled", "favors": "product_a"},
            {"feature": "total_events", "favors": "product_b"},
        ],
        "Phone A",
        "Phone B",
    )

    assert "listed price favored Phone A" in summary
    assert "category activity in this dataset favored Phone B" in summary
