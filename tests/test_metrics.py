"""
tests/test_metrics.py
=======================
Unit tests for the ranking/NLP/explainability evaluation metrics.
"""
from evaluation.metrics import (
    bleu_score, dcg_at_k, f1_at_k, mean_average_precision, ndcg_at_k,
    precision_at_k, recall_at_k, rouge_l,
)


def test_precision_and_recall_at_k_perfect_ranking():
    labels = [1, 1, 1, 0, 0]
    assert precision_at_k(labels, 3) == 1.0
    assert recall_at_k(labels, 3) == 1.0
    assert f1_at_k(labels, 3) == 1.0


def test_ndcg_at_k_is_one_for_ideal_ranking():
    labels = [1, 1, 0, 0]
    assert ndcg_at_k(labels, 4) == 1.0


def test_map_across_multiple_users():
    lists = [[1, 0, 1], [0, 1, 0]]
    score = mean_average_precision(lists)
    assert 0.0 <= score <= 1.0


def test_bleu_and_rouge_identical_strings_score_high():
    text = "this product ranked highly due to price and category"
    assert bleu_score(text, text) > 0.9
    assert rouge_l(text, text) > 0.9
