"""
ranking/train.py
==================
End-to-end training entry point:
  1. loads processed data + graph features
  2. builds the feature matrix
  3. optionally runs k-fold cross-validation and grid-search hyperparameter
     tuning for the LightGBM base model (representative of the tuning
     workflow; the same `tune_lightgbm` pattern applies to XGBoost)
  4. trains the final hybrid ensemble on the full training split
  5. persists the trained ensemble + feature list to disk for serving
"""
from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common.config import get_config, get_logger
from evaluation.metrics import ndcg_at_k
from feature_engineering.features import FeatureEngineer, TARGET_PROXY_FEATURES
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from preprocessing.data_loader import DatasetLoader
from preprocessing.preprocess import Preprocessor, split_train_val_test
from ranking.ensemble import HybridRankingEnsemble

logger = get_logger(__name__)

LABEL_COL = "label"
ID_COLS = ["user_id", "product_id"]


def _feature_columns(matrix: pd.DataFrame) -> List[str]:
    """Select model inputs without IDs, labels, or direct target proxies."""
    excluded = set(ID_COLS + [LABEL_COL]) | TARGET_PROXY_FEATURES
    if get_config()["data"]["dataset"] == "flipkart":
        excluded.update({"price_scaled", "avg_rating_scaled"})
    return [
        column for column in matrix.columns
        if column not in excluded
    ]


def build_features_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run loading -> preprocessing -> KG features -> feature engineering, returning train/val/test matrices."""
    raw = DatasetLoader().load()
    processed = Preprocessor().run(raw)

    kg_builder = KnowledgeGraphBuilder()
    kg_builder.build(processed.users, processed.products, processed.interactions, processed.reviews)
    graph_feats = kg_builder.compute_graph_features()
    kg_embeddings = kg_builder.compute_graph_embeddings()
    graph_feats = graph_feats.join(kg_embeddings, how="left").fillna(0.0)

    fe = FeatureEngineer(graph_features=graph_feats)
    full_matrix = fe.build_full_feature_matrix(
        processed.users, processed.products, processed.interactions, processed.reviews
    )

    train_int, val_int, test_int = split_train_val_test(processed.interactions)

    def _subset(matrix: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
        keys = interactions[["user_id", "product_id"]].drop_duplicates()
        return matrix.merge(keys, on=["user_id", "product_id"], how="inner")

    return _subset(full_matrix, train_int), _subset(full_matrix, val_int), _subset(full_matrix, test_int)


def tune_lightgbm(
    X: pd.DataFrame, y: pd.Series, group_ids: pd.Series, param_grid: Dict[str, List], n_splits: int = 3
) -> Dict:
    """
    Grid-search + GroupKFold cross-validation for LightGBM hyperparameters,
    scored by NDCG@10. Demonstrates the tuning workflow; XGBoost / neural
    models follow the identical pattern (grid or random search + CV).

    Args:
        X, y, group_ids: Training data and per-row user_id groups.
        param_grid: Dict of hyperparameter name -> list of candidate values.
        n_splits: Number of GroupKFold splits.

    Returns:
        The best-performing hyperparameter combination found.
    """
    import lightgbm as lgb

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    best_score, best_params = -np.inf, None

    gkf = GroupKFold(n_splits=min(n_splits, group_ids.nunique()))
    for combo in combinations:
        params = dict(zip(keys, combo))
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=group_ids):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            g_train = group_ids.iloc[train_idx]
            g_train_sizes = g_train.value_counts(sort=False).reindex(g_train.unique()).tolist()

            model = lgb.LGBMRanker(objective="lambdarank", verbosity=-1, **params)
            model.fit(X_train, y_train, group=g_train_sizes)
            preds = model.predict(X_val)

            val_groups = group_ids.iloc[val_idx]
            score = _mean_ndcg_by_group(preds, y_val.values, val_groups, k=10)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        logger.info("Tuning combo %s -> mean NDCG@10=%.4f", params, mean_score)
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    logger.info("Best LightGBM params: %s (NDCG@10=%.4f)", best_params, best_score)
    return best_params


def _mean_ndcg_by_group(preds: np.ndarray, labels: np.ndarray, groups: pd.Series, k: int) -> float:
    scores = []
    df = pd.DataFrame({"pred": preds, "label": labels, "group": groups.values})
    for _, g in df.groupby("group"):
        ranked = g.sort_values("pred", ascending=False)["label"].tolist()
        scores.append(ndcg_at_k(ranked, k))
    return float(np.mean(scores)) if scores else 0.0


def train_and_save(
    output_path: str = "results/hybrid_ensemble.pkl", run_tuning: bool = False
) -> Dict[str, int]:
    """
    Full training entry point: builds features, optionally tunes
    hyperparameters, trains the hybrid ensemble, and pickles it to disk.

    Args:
        output_path: Where to persist the trained ensemble artifact.
        run_tuning: If True, runs `tune_lightgbm` before final training
            (slower; illustrates the hyperparameter-search workflow).

    Returns:
        Counts for the train, validation, and test feature matrices.
    """
    train_df, val_df, test_df = build_features_pipeline()
    feature_cols = _feature_columns(train_df)

    X_train, y_train, g_train = train_df[feature_cols], train_df[LABEL_COL], train_df["user_id"]

    if run_tuning:
        best_params = tune_lightgbm(
            X_train, y_train, g_train,
            param_grid={"num_leaves": [31, 63], "learning_rate": [0.05, 0.1], "n_estimators": [100, 300]},
        )
        cfg = get_config()
        cfg["ranking"]["models"]["lightgbm"].update(best_params)

    ensemble = HybridRankingEnsemble()
    ensemble.fit(X_train, y_train, g_train)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"ensemble": ensemble, "feature_cols": feature_cols}, fh)
    logger.info("Saved trained hybrid ensemble to %s", out)

    if not val_df.empty:
        X_val, y_val, g_val = val_df[feature_cols], val_df[LABEL_COL], val_df["user_id"]
        val_preds = ensemble.predict(X_val, group_ids=g_val)
        val_ndcg = _mean_ndcg_by_group(val_preds, y_val.values, g_val, k=10)
        logger.info("Validation NDCG@10: %.4f", val_ndcg)
    return {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "user_groups": int(train_df["user_id"].nunique()),
        "products_in_train": int(pd.concat(
            [train_df["product_id"], val_df["product_id"], test_df["product_id"]]
        ).nunique()),
        "feature_count": len(feature_cols),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hybrid ranking ensemble")
    parser.add_argument("--output", default="results/hybrid_ensemble.pkl")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning before final training")
    args = parser.parse_args()
    train_and_save(output_path=args.output, run_tuning=args.tune)
