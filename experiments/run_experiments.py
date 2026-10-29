"""
experiments/run_experiments.py
================================
Runs the experimental suite required for the research report:
  1. Baseline models (popularity, random) vs the hybrid ensemble
  2. Ablation study: hybrid ensemble with each base model removed in turn
  3. Statistical significance testing (paired t-test) between the full
     hybrid ensemble and each ablated / baseline variant on NDCG@10
  4. Publication-quality comparison plot

Results are written to results/experiments_report.json and
results/ablation_comparison.png.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common.config import get_config, get_logger
from evaluation.metrics import ndcg_at_k
from ranking.ensemble import HybridRankingEnsemble
from ranking.train import LABEL_COL, build_features_pipeline

logger = get_logger(__name__)


def _per_user_ndcg(df: pd.DataFrame, score_col: str, k: int = 10) -> List[float]:
    scores = []
    for _, group in df.groupby("user_id"):
        ranked = group.sort_values(score_col, ascending=False)[LABEL_COL].tolist()
        scores.append(ndcg_at_k(ranked, k))
    return scores


def _train_variant(train_df: pd.DataFrame, feature_cols: List[str], enabled_models: List[str]) -> HybridRankingEnsemble:
    cfg = get_config()
    original_flags = {name: cfg["ranking"]["models"][name]["enabled"] for name in ["lightgbm", "xgboost", "neural_ranker"]}
    for name in original_flags:
        cfg["ranking"]["models"][name]["enabled"] = name in enabled_models

    ensemble = HybridRankingEnsemble()
    ensemble.fit(train_df[feature_cols], train_df[LABEL_COL], train_df["user_id"])

    for name, flag in original_flags.items():  # restore config for subsequent runs
        cfg["ranking"]["models"][name]["enabled"] = flag
    return ensemble


def run_ablation_study(output_dir: str = "results") -> Dict:
    """
    Train the hybrid ensemble with every base model enabled, then with
    each base model removed in turn, evaluate NDCG@10 for each variant on
    the test split, and run paired t-tests against the full ensemble.

    Args:
        output_dir: Where to write the JSON report and comparison plot.

    Returns:
        The full ablation report as a dict.
    """
    train_df, _, test_df = build_features_pipeline()
    feature_cols = [c for c in train_df.columns if c not in ["user_id", "product_id", LABEL_COL]]
    base_model_names = ["lightgbm", "xgboost", "neural_ranker"]

    variants = {"full_ensemble": base_model_names}
    for name in base_model_names:
        remaining = [m for m in base_model_names if m != name]
        if remaining:
            variants[f"without_{name}"] = remaining
    variants["random_baseline"] = []  # special-cased below (no trained model)

    results: Dict[str, Dict] = {}
    ndcg_distributions: Dict[str, List[float]] = {}

    for variant_name, enabled_models in variants.items():
        if variant_name == "random_baseline":
            rng = np.random.default_rng(get_config()["project"]["seed"])
            scored = test_df.copy()
            scored["score"] = rng.random(len(scored))
        else:
            ensemble = _train_variant(train_df, feature_cols, enabled_models)
            scored = test_df.copy()
            scored["score"] = ensemble.predict(scored[feature_cols])

        per_user_ndcg = _per_user_ndcg(scored, "score", k=10)
        ndcg_distributions[variant_name] = per_user_ndcg
        results[variant_name] = {"mean_ndcg_at_10": float(np.mean(per_user_ndcg)) if per_user_ndcg else 0.0}
        logger.info("Variant '%s' -> mean NDCG@10=%.4f", variant_name, results[variant_name]["mean_ndcg_at_10"])

    full_dist = ndcg_distributions["full_ensemble"]
    for variant_name, dist in ndcg_distributions.items():
        if variant_name == "full_ensemble" or not dist or not full_dist:
            continue
        n = min(len(dist), len(full_dist))
        t_stat, p_value = stats.ttest_rel(full_dist[:n], dist[:n])
        results[variant_name]["paired_ttest_vs_full"] = {
            "t_statistic": float(t_stat), "p_value": float(p_value), "significant_at_0.05": bool(p_value < 0.05),
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "experiments_report.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _plot_ablation(results, out / "ablation_comparison.png")
    logger.info("Ablation study report written to %s", out / "experiments_report.json")
    return results


def _plot_ablation(results: Dict, output_path: Path) -> None:
    names = list(results.keys())
    values = [results[n]["mean_ndcg_at_10"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, values, color="steelblue")
    ax.set_ylabel("Mean NDCG@10")
    ax.set_title("Ablation Study: Hybrid Ensemble Component Contributions")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_ablation_study()
