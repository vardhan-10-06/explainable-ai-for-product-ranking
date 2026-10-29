"""
tests/test_evaluation.py
==========================
Tests for evaluation-report generation helpers (unit-level; the full
evaluate_model() integration path requires a trained ensemble artifact
and is exercised in the demo notebook / CI integration stage instead).
"""
import pandas as pd

from evaluation.evaluate import (
    _expected_top_k_catalog_metrics,
    _category_brand_metrics,
    _mean_reciprocal_rank_with_ties,
    _popularity_baseline_scores,
    _ranking_report,
)
from evaluation.metrics import measure_latency, measure_memory_usage


def test_measure_latency_returns_positive_float():
    def dummy_fn():
        return sum(range(1000))
    latency = measure_latency(dummy_fn)
    assert latency >= 0.0


def test_measure_memory_usage_returns_positive_float():
    def dummy_fn():
        return [i for i in range(10000)]
    mem = measure_memory_usage(dummy_fn)
    assert mem >= 0.0


def test_mrr_averages_equally_scored_items_instead_of_using_row_order():
    first_order = pd.DataFrame({
        "user_id": ["category", "category"],
        "label": [1, 0],
        "score": [0.0, 0.0],
    })
    reverse_order = first_order.iloc[::-1].reset_index(drop=True)

    assert _mean_reciprocal_rank_with_ties(first_order, "score") == 0.75
    assert _mean_reciprocal_rank_with_ties(reverse_order, "score") == 0.75


def test_all_ranking_metrics_average_over_score_ties():
    tied_rows = pd.DataFrame({
        "user_id": ["category", "category"],
        "product_id": ["positive", "negative"],
        "label": [1, 0],
        "score": [0.0, 0.0],
    })
    reversed_rows = tied_rows.iloc[::-1].reset_index(drop=True)
    metrics = _ranking_report(tied_rows, "score", [1])
    reversed_metrics = _ranking_report(reversed_rows, "score", [1])

    for key in ("precision_at_1", "recall_at_1", "f1_at_1", "ndcg_at_1", "map", "mrr"):
        assert metrics[key] == reversed_metrics[key]
    assert metrics["precision_at_1"] == 0.5
    assert metrics["ndcg_at_1"] == 0.5
    assert metrics["map"] == 0.75
    assert metrics["mrr"] == 0.75


def test_roc_auc_is_null_when_evaluation_rows_have_one_relevance_class():
    rows = pd.DataFrame({
        "user_id": ["category", "category"],
        "product_id": ["first", "second"],
        "label": [1, 1],
        "score": [0.9, 0.1],
    })

    assert _ranking_report(rows, "score", [1])["roc_auc"] is None


def test_coverage_and_novelty_do_not_use_csv_order_to_break_top_k_ties():
    tied_rows = pd.DataFrame({
        "user_id": ["category"] * 12,
        "product_id": [f"p{index}" for index in range(12)],
        "label": [index % 2 for index in range(12)],
        "score": [0.0] * 12,
    })

    first = _expected_top_k_catalog_metrics(tied_rows, "score", k=10)
    reversed_result = _expected_top_k_catalog_metrics(
        tied_rows.iloc[::-1].reset_index(drop=True), "score", k=10
    )

    assert first == reversed_result
    assert first[0] == 10 / 12


def test_popularity_baseline_uses_training_counts_not_test_data():
    train = pd.DataFrame({"product_id": ["a", "a", "b"]})
    test = pd.DataFrame({"product_id": ["a", "b", "unseen"]})

    scores = _popularity_baseline_scores(train, test)

    assert scores.tolist() == [2.0, 1.0, 0.0]


def test_category_brand_metrics_are_scoped_to_category_and_brand():
    rows = pd.DataFrame({
        "user_id": ["mobiles", "mobiles", "mobiles", "mobiles", "laptops", "laptops"],
        "brand_key": ["acme", "acme", "other", "other", "acme", "acme"],
        "product_id": ["m1", "m2", "m3", "m4", "l1", "l2"],
        "label": [1, 0, 1, 0, 1, 0],
        "score": [0.9, 0.1, 0.8, 0.2, 0.8, 0.2],
    })

    metrics = _category_brand_metrics(rows, [10], "score")

    assert set(metrics["mobiles"]) == {"acme", "other"}
    assert metrics["mobiles"]["acme"]["test_rows"] == 2
    assert metrics["mobiles"]["acme"]["products"] == 2
    assert metrics["mobiles"]["acme"]["metrics"]["ndcg_at_10"] == 1.0
    assert metrics["mobiles"]["other"]["test_rows"] == 2
    assert set(metrics["laptops"]) == {"acme"}
