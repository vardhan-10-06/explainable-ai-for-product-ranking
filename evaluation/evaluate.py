"""
evaluation/evaluate.py
========================
Runs the full evaluation suite over a trained ensemble and its held-out
test set, producing a JSON report and comparison plots (baseline vs
hybrid ensemble) under results/.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import get_config, get_logger
from evaluation.metrics import (
    measure_latency, measure_memory_usage, roc_auc,
)
from preprocessing.data_loader import DatasetLoader
from ranking.train import LABEL_COL, build_features_pipeline

logger = get_logger(__name__)


def evaluate_model(ensemble_path: str = "results/hybrid_ensemble.pkl", output_dir: str = "results") -> Dict:
    """
    Run the full evaluation suite (ranking + efficiency metrics) on the
    test split, comparing the hybrid ensemble against a naive popularity
    baseline, and write a JSON report + comparison plot to `output_dir`.

    Args:
        ensemble_path: Path to the pickled trained ensemble (see ranking/train.py).
        output_dir: Directory to write the evaluation report and plots to.

    Returns:
        The full evaluation report as a dict (also written to disk).
    """
    cfg = get_config()["evaluation"]
    k_values = cfg["k_values"]

    with open(ensemble_path, "rb") as fh:
        artifact = pickle.load(fh)
    ensemble, feature_cols = artifact["ensemble"], artifact["feature_cols"]

    train_df, _, test_df = build_features_pipeline()
    if test_df.empty:
        raise RuntimeError("Test split is empty -- check dataset size / min-interaction filters in config.")

    X_test = test_df[feature_cols]
    test_df = test_df.copy()
    test_df["ensemble_score"] = ensemble.predict(X_test, group_ids=test_df["user_id"])
    test_df["baseline_score"] = _popularity_baseline_scores(train_df, test_df)

    report = {
        "hybrid_ensemble": _ranking_report(test_df, "ensemble_score", k_values),
        "popularity_baseline": _ranking_report(test_df, "baseline_score", k_values),
        "efficiency": _efficiency_report(ensemble, X_test),
        "metric_notes": {
            "tie_handling": (
                "Ranking metrics average over all possible orderings of products with "
                "equal scores, including top-k catalog coverage and novelty, so file "
                "row order cannot change ranking quality."
            ),
            "popularity_baseline": (
                "Product interaction counts are calculated from the training split only; "
                "unseen products receive tied zero scores."
            ),
            "roc_auc": (
                "ROC-AUC measures discrimination, not classification accuracy; a score "
                "of 0.5 indicates chance-level ordering. It is null when the evaluated "
                "rows contain only one relevance class."
            ),
        },
    }
    if get_config()["data"]["dataset"] == "flipkart":
        brand_map = (
            DatasetLoader().load().products.set_index("product_id")["brand"]
            .fillna("").astype(str).str.strip().str.lower()
        )
        test_df["brand_key"] = test_df["product_id"].map(brand_map).fillna("")
        report["category_metrics"] = _category_metrics(
            test_df, k_values, score_col="ensemble_score"
        )
        report["category_baseline_metrics"] = _category_metrics(
            test_df, k_values, score_col="baseline_score"
        )
        report["category_brand_metrics"] = _category_brand_metrics(
            test_df, k_values, score_col="ensemble_score"
        )
        report["category_brand_baseline_metrics"] = _category_brand_metrics(
            test_df, k_values, score_col="baseline_score"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)

    _plot_comparison(report, out / "ranking_comparison.png")
    logger.info("Evaluation report written to %s", out / "evaluation_report.json")
    return report


def _ranking_report(df: pd.DataFrame, score_col: str, k_values: list) -> Dict:
    metrics: Dict = {}
    for k in k_values:
        group_scores = [
            _expected_ranking_metrics(group, score_col, k)
            for _, group in df.groupby("user_id")
        ]
        metrics[f"precision_at_{k}"] = float(np.mean([
            group_metrics["precision"] for group_metrics in group_scores
        ]))
        metrics[f"recall_at_{k}"] = float(np.mean([
            group_metrics["recall"] for group_metrics in group_scores
        ]))
        metrics[f"f1_at_{k}"] = float(np.mean([
            group_metrics["f1"] for group_metrics in group_scores
        ]))
        metrics[f"ndcg_at_{k}"] = float(np.mean([
            group_metrics["ndcg"] for group_metrics in group_scores
        ]))

    group_average_precisions = [
        _expected_average_precision(group, score_col)
        for _, group in df.groupby("user_id")
    ]
    metrics["map"] = float(np.mean(group_average_precisions)) if group_average_precisions else 0.0
    metrics["mrr"] = _mean_reciprocal_rank_with_ties(df, score_col)
    if df[LABEL_COL].nunique() < 2:
        metrics["roc_auc"] = None
    else:
        auc = roc_auc(df[LABEL_COL].values, df[score_col].values)
        metrics["roc_auc"] = auc if math.isfinite(auc) else None

    metrics["coverage"], metrics["novelty"] = _expected_top_k_catalog_metrics(
        df, score_col, k=10
    )

    return metrics


def _expected_top_k_catalog_metrics(
    df: pd.DataFrame, score_col: str, k: int
) -> tuple[float, float]:
    """Compute catalog coverage and novelty without arbitrarily breaking score ties."""
    catalog_size = int(df["product_id"].nunique())
    if not catalog_size:
        return 0.0, 0.0

    no_recommend_probability: Dict[str, float] = {}
    product_counts = df["product_id"].value_counts()
    expected_novelty_sum = 0.0
    expected_recommendation_count = 0

    for _, group in df.groupby("user_id"):
        expected_recommendation_count += min(k, len(group))
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort")
        for start, size, _ in _score_tie_blocks(group, score_col):
            overlap = max(0, min(start + size, k) - start)
            if overlap == 0:
                continue
            inclusion_probability = overlap / size
            product_ids = ordered.iloc[start:start + size]["product_id"]
            for product_id in product_ids:
                key = str(product_id)
                no_recommend_probability[key] = (
                    no_recommend_probability.get(key, 1.0)
                    * (1.0 - inclusion_probability)
                )
                probability = float(product_counts[product_id] / len(df))
                expected_novelty_sum += inclusion_probability * -math.log2(
                    max(probability, 1e-6)
                )

    expected_covered_items = sum(
        1.0 - probability_none
        for probability_none in no_recommend_probability.values()
    )
    expected_novelty = (
        expected_novelty_sum / expected_recommendation_count
        if expected_recommendation_count else 0.0
    )
    return expected_covered_items / catalog_size, expected_novelty


def _score_tie_blocks(group: pd.DataFrame, score_col: str) -> list[tuple[int, int, int]]:
    """Return (zero-based start, size, relevant count) for equal-score blocks."""
    ordered = group.sort_values(score_col, ascending=False, kind="mergesort")
    scores = ordered[score_col].to_numpy()
    labels = ordered[LABEL_COL].to_numpy(dtype=int)
    blocks = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores[end] == scores[start]:
            end += 1
        blocks.append((start, end - start, int(labels[start:end].sum())))
        start = end
    return blocks


def _expected_ranking_metrics(
    group: pd.DataFrame, score_col: str, k: int
) -> Dict[str, float]:
    """Compute expected top-k metrics over all permutations within score ties."""
    labels = group[LABEL_COL].to_numpy(dtype=int)
    item_count = len(labels)
    positive_count = int(labels.sum())
    cutoff = min(k, item_count)
    if cutoff == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0}

    expected_hits = 0.0
    expected_dcg = 0.0
    for start, size, relevant in _score_tie_blocks(group, score_col):
        overlap = max(0, min(start + size, cutoff) - start)
        if overlap == 0:
            continue
        expected_relevance = relevant / size
        expected_hits += overlap * expected_relevance
        ranks = np.arange(start + 1, start + overlap + 1, dtype=float)
        expected_dcg += expected_relevance * float(
            np.sum(1.0 / np.log2(ranks + 1))
        )

    ideal_relevance = np.sort(labels)[::-1][:cutoff]
    ideal_ranks = np.arange(1, len(ideal_relevance) + 1, dtype=float)
    ideal_dcg = float(np.sum(ideal_relevance / np.log2(ideal_ranks + 1)))
    precision = expected_hits / cutoff
    recall = expected_hits / positive_count if positive_count else 0.0
    f1 = (
        2.0 * expected_hits / (cutoff + positive_count)
        if positive_count
        else 0.0
    )
    ndcg = expected_dcg / ideal_dcg if ideal_dcg else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "ndcg": ndcg}


def _expected_average_precision(group: pd.DataFrame, score_col: str) -> float:
    """Compute expected AP over all permutations within equal-score blocks."""
    positive_count = int(group[LABEL_COL].sum())
    if positive_count == 0:
        return 0.0

    hits_before_block = 0
    expected_precision_sum = 0.0
    for start, size, relevant in _score_tie_blocks(group, score_col):
        if relevant:
            for position in range(1, size + 1):
                expected_previous_hits = (
                    (position - 1) * (relevant - 1) / (size - 1)
                    if size > 1 else 0.0
                )
                expected_precision_sum += (
                    (relevant / size)
                    * (hits_before_block + 1 + expected_previous_hits)
                    / (start + position)
                )
        hits_before_block += relevant
    return expected_precision_sum / positive_count


def _mean_reciprocal_rank_with_ties(df: pd.DataFrame, score_col: str) -> float:
    """Compute expected MRR across all orderings of equally scored items."""
    group_scores = []
    for _, group in df.groupby("user_id"):
        blocks = _score_tie_blocks(group, score_col)
        reciprocal_rank = 0.0

        for start, tie_size, relevant_count in blocks:
            if relevant_count:
                total_orders = math.comb(tie_size, relevant_count)
                reciprocal_rank = sum(
                    math.comb(tie_size - position, relevant_count - 1)
                    / total_orders
                    / (start + position)
                    for position in range(1, tie_size - relevant_count + 2)
                )
                break

        group_scores.append(reciprocal_rank)
    return float(np.mean(group_scores)) if group_scores else 0.0


def _popularity_baseline_scores(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.Series:
    """Score test products using training-split interaction counts only."""
    training_counts = train_df["product_id"].value_counts()
    return test_df["product_id"].map(training_counts).fillna(0).astype(float)


def _category_metrics(
    test_df: pd.DataFrame, k_values: list, score_col: str
) -> Dict:
    """Report held-out ranking metrics independently for each Flipkart category."""
    return {
        str(category).strip().lower(): {
            "metrics": _ranking_report(category_df, score_col, k_values),
            "test_rows": len(category_df),
            "products": int(category_df["product_id"].nunique()),
        }
        for category, category_df in test_df.groupby("user_id")
    }


def _category_brand_metrics(
    test_df: pd.DataFrame, k_values: list, score_col: str
) -> Dict:
    """Report held-out metrics for each category and normalized product brand."""
    return {
        str(category).strip().lower(): {
            str(brand).strip().lower(): {
                "metrics": _ranking_report(brand_group, score_col, k_values),
                "test_rows": len(brand_group),
                "products": int(brand_group["product_id"].nunique()),
            }
            for brand, brand_group in category_group.groupby("brand_key")
            if str(brand).strip()
        }
        for category, category_group in test_df.groupby("user_id")
    }


def _efficiency_report(ensemble, X_test: pd.DataFrame) -> Dict:
    sample = X_test.iloc[: min(100, len(X_test))]
    latency = measure_latency(ensemble.predict, sample)
    memory_mb = measure_memory_usage(ensemble.predict, sample)
    return {
        "latency_seconds_per_100_rows": latency,
        "peak_memory_mb": memory_mb,
        "rows_scored": len(sample),
    }


def _plot_comparison(report: Dict, output_path: Path) -> None:
    metrics_to_plot = ["ndcg_at_10", "map", "mrr", "precision_at_10"]
    hybrid_vals = [report["hybrid_ensemble"].get(m, 0.0) for m in metrics_to_plot]
    baseline_vals = [report["popularity_baseline"].get(m, 0.0) for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, hybrid_vals, width, label="Hybrid Ensemble")
    ax.bar(x + width / 2, baseline_vals, width, label="Popularity Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=20)
    ax.set_ylabel("Score")
    ax.set_title("Ranking Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    evaluate_model()
