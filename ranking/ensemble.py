"""
ranking/ensemble.py
=====================
Combines the individual ranking models (LightGBM, XGBoost, NeuralRanker,
DeepFM) into a single hybrid ranker.

Supported ensemble strategies (config: ranking.ensemble.strategy):
  * "weighted"      - weighted sum of raw model scores (weights in config).
                       Simple, fast, and works well when models are already
                       roughly well-calibrated against each other.
  * "rank_average"   - average of each model's *rank* (not raw score) per
                       user. More robust when models' score scales differ
                       wildly (trees vs. neural nets), at the cost of
                       discarding score magnitude information.
  * "stacking"        - trains a light logistic-regression meta-learner on
                       top of the base models' out-of-fold predictions.
                       Highest potential accuracy, more complex and needs
                       more data to avoid overfitting the meta-learner.

Default is "weighted" (see config) as the most robust option for a
research-scale dataset; the trade-offs above are surfaced in the docs so a
practitioner can switch strategies for their own data scale.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from common.config import get_config, get_logger
from ranking.models import DeepFMRanker, LightGBMRanker, NeuralRanker, XGBoostRanker

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "lightgbm": LightGBMRanker,
    "xgboost": XGBoostRanker,
    "neural_ranker": NeuralRanker,
    "deepfm": DeepFMRanker,
}


class HybridRankingEnsemble:
    """Trains and combines multiple base ranking models into one hybrid ranker."""

    def __init__(self):
        self.cfg = get_config()["ranking"]
        self.models: Dict[str, object] = {}
        self.strategy = self.cfg["ensemble"]["strategy"]
        self.weights = self.cfg["ensemble"]["weights"]
        self.meta_learner: LogisticRegression | None = None
        self.feature_names_: List[str] = []
        self.score_statistics_: Dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series) -> "HybridRankingEnsemble":
        """
        Train every enabled base model, then (if strategy == "stacking")
        train a meta-learner on their out-of-fold predictions.

        Args:
            X: Feature matrix.
            y: Binary relevance labels (1 = purchased/relevant).
            group_ids: Per-row user_id, used as the ranking "query group".
        """
        self.feature_names_ = list(X.columns)
        model_cfgs = self.cfg["models"]

        for name, cfg_key in [
            ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
            ("neural_ranker", "neural_ranker"),
        ]:
            if model_cfgs.get(cfg_key, {}).get("enabled", False):
                logger.info("Training base model: %s", name)
                self.models[name] = MODEL_REGISTRY[name]().fit(X, y, group_ids)

        # DeepFM shares the neural_ranker config block (architecture knobs) but
        # is registered under its own key so it can be toggled independently
        # if the user adds a "deepfm.enabled" flag to config in the future.
        if model_cfgs.get("neural_ranker", {}).get("enabled", False):
            logger.info("Training base model: deepfm")
            self.models["deepfm"] = MODEL_REGISTRY["deepfm"]().fit(X, y, group_ids)

        if self.strategy == "weighted":
            self.calibrate(X)
        if self.strategy == "stacking":
            self._fit_meta_learner(X, y, group_ids)

        logger.info("Hybrid ensemble trained with models: %s (strategy=%s)", list(self.models), self.strategy)
        return self

    def calibrate(self, X: pd.DataFrame) -> None:
        """Fix per-model score scales so predictions remain comparable across requests."""
        self.score_statistics_ = {}
        for name, model in self.models.items():
            scores = np.asarray(model.predict(X), dtype=float)
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            self.score_statistics_[name] = (mean, max(std, 1e-8))

    def _fit_meta_learner(self, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series) -> None:
        """Build out-of-fold base-model predictions and fit a logistic-regression meta-learner on top."""
        gkf = GroupKFold(n_splits=min(5, group_ids.nunique()) if group_ids.nunique() >= 2 else 2)
        oof_preds = {name: np.zeros(len(X)) for name in self.models}

        for train_idx, val_idx in gkf.split(X, y, groups=group_ids):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, g_train = y.iloc[train_idx], group_ids.iloc[train_idx]
            for name, cls in [(n, type(m)) for n, m in self.models.items()]:
                fold_model = cls().fit(X_train, y_train, g_train)
                oof_preds[name][val_idx] = fold_model.predict(X_val)

        meta_X = pd.DataFrame(oof_preds)
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.meta_learner.fit(meta_X, y)
        logger.info("Meta-learner (stacking) fitted on out-of-fold base predictions.")

    def _finalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize output to a stable, deterministic float representation.

        The ensemble combines several ML models whose independent numerical
        pipelines can differ by tiny floating-point amounts between equivalent
        calls (for example when a single row is scored in isolation vs. as part of
        a larger batch). Rounding keeps the ranking signal intact while ensuring
        repeated calls with identical inputs yield identical output values.
        """
        output = np.asarray(scores, dtype=np.float64)
        return np.round(output, 5)

    def predict(self, X: pd.DataFrame, group_ids: pd.Series | None = None) -> np.ndarray:
        """
        Produce final ensemble relevance scores for X.

        Args:
            X: Feature matrix to score.
            group_ids: Required only for the "rank_average" strategy (needs
                to know which rows belong to the same user to compute
                within-group ranks).

        Returns:
            1D array of ensemble scores, higher = more relevant.
        """
        base_scores = {name: model.predict(X) for name, model in self.models.items()}

        if self.strategy == "weighted":
            return self._finalize_scores(self._weighted_combine(base_scores))
        if self.strategy == "rank_average":
            return self._finalize_scores(self._rank_average_combine(base_scores, group_ids))
        if self.strategy == "stacking":
            if self.meta_learner is None:
                raise RuntimeError("Stacking strategy selected but meta-learner was not fitted.")
            meta_X = pd.DataFrame(base_scores)
            return self._finalize_scores(self.meta_learner.predict_proba(meta_X)[:, 1])
        raise ValueError(f"Unknown ensemble strategy: {self.strategy}")

    def _weighted_combine(self, base_scores: Dict[str, np.ndarray]) -> np.ndarray:
        combined = np.zeros(len(next(iter(base_scores.values()))))
        total_weight = 0.0
        score_statistics = getattr(self, "score_statistics_", {})
        for name, scores in base_scores.items():
            weight = self.weights.get(name, 1.0 / len(base_scores))
            if name in score_statistics:
                mean, std = score_statistics[name]
            else:
                mean, std = float(scores.mean()), float(scores.std()) + 1e-8
            normalized = (scores - mean) / max(std, 1e-8)
            combined += weight * normalized
            total_weight += weight
        return combined / max(total_weight, 1e-8)

    def _rank_average_combine(self, base_scores: Dict[str, np.ndarray], group_ids: pd.Series | None) -> np.ndarray:
        if group_ids is None:
            raise ValueError("rank_average strategy requires group_ids to be passed to predict().")
        ranks = pd.DataFrame(base_scores)
        ranks["group"] = group_ids.values
        rank_cols = []
        for name in base_scores:
            col = f"{name}_rank"
            ranks[col] = ranks.groupby("group")[name].rank(ascending=False)
            rank_cols.append(col)
        avg_rank = ranks[rank_cols].mean(axis=1)
        return -avg_rank.values  # lower rank number = better, so negate for "higher = better" convention

    def get_base_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return each base model's raw score per row -- used by the explainability module."""
        return pd.DataFrame({name: model.predict(X) for name, model in self.models.items()})

    def get_feature_importances(self) -> pd.DataFrame:
        """Aggregate feature importances across all trained base models into one table."""
        importances = {}
        for name, model in self.models.items():
            try:
                importances[name] = model.feature_importance()
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not extract feature importance from %s: %s", name, exc)
        df = pd.DataFrame(importances).fillna(0.0)
        if df.empty:
            return df
        # Normalize each model's importances to [0, 1] before averaging so no
        # single model's raw scale dominates the aggregate column.
        normalized = df / df.max().replace(0, 1)
        df["ensemble_avg"] = normalized.mean(axis=1)
        return df.sort_values("ensemble_avg", ascending=False)
