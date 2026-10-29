"""
explainability/shap_explainer.py
===================================
SHAP-based explainability for the hybrid ranking ensemble.

Since SHAP's TreeExplainer only works on a single tree model (not an
ensemble-of-heterogeneous-models), we compute SHAP values per base model
that supports it natively (LightGBM, XGBoost via TreeExplainer) and use
KernelExplainer as a model-agnostic fallback for the neural/DeepFM
components. The final "ensemble SHAP" is the weight-adjusted sum of each
base model's SHAP values -- valid because SHAP values are additive and the
ensemble's weighted-sum strategy is linear in the base scores.

RankSHAP: rather than depending on an unmaintained external `rankshap`
package, we implement the standard rank-explanation adapter described in
the RankSHAP literature: compute SHAP values for two candidate items and
attribute the *difference* in their scores (i.e. why A ranked above B) to
the features whose SHAP contributions differ most between the two.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from common.config import get_config, get_logger
from ranking.ensemble import HybridRankingEnsemble

logger = get_logger(__name__)


class ShapRankingExplainer:
    """Computes global and local SHAP-based explanations for the hybrid ensemble."""

    def __init__(self, ensemble: HybridRankingEnsemble, background_data: pd.DataFrame):
        """
        Args:
            ensemble: A fitted HybridRankingEnsemble.
            background_data: A representative sample of training features,
                used as the SHAP background/reference distribution.
        """
        self.ensemble = ensemble
        self.cfg = get_config()["explainability"]["shap"]
        n_bg = min(self.cfg["background_samples"], len(background_data))
        self.background = background_data.sample(n=n_bg, random_state=get_config()["project"]["seed"])
        self._tree_explainers: Dict[str, shap.TreeExplainer] = {}
        self._kernel_explainers: Dict[str, shap.KernelExplainer] = {}
        self._build_explainers()

    def _build_explainers(self) -> None:
        for name, model in self.ensemble.models.items():
            if name in ("lightgbm", "xgboost"):
                try:
                    self._tree_explainers[name] = shap.TreeExplainer(model.model)
                except Exception as exc:  # pragma: no cover
                    logger.warning("TreeExplainer failed for %s (%s); falling back to KernelExplainer.", name, exc)
                    self._kernel_explainers[name] = shap.KernelExplainer(model.predict, self.background)
            else:
                # Neural / DeepFM: use a small Kernel explainer (model-agnostic, slower).
                self._kernel_explainers[name] = shap.KernelExplainer(model.predict, self.background)

    def explain_instances(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the ensemble-weighted SHAP values for every row in X.

        Args:
            X: Feature rows to explain (same columns as training data).

        Returns:
            DataFrame of shape (len(X), n_features) with the combined,
            weight-adjusted SHAP contribution of every feature to every row.
        """
        combined = np.zeros((len(X), X.shape[1]))
        model_weights = {}
        weights = self.ensemble.weights if self.ensemble.strategy == "weighted" else {
            name: 1.0 / len(self.ensemble.models) for name in self.ensemble.models
        }
        score_statistics = getattr(self.ensemble, "score_statistics_", {})
        for name in self.ensemble.models:
            model_weights[name] = weights.get(name, 1.0 / len(self.ensemble.models))
        total_model_weight = sum(model_weights.values())

        for name in self.ensemble.models:
            weight = model_weights[name]
            if name in self._tree_explainers:
                shap_values = self._tree_explainers[name].shap_values(X)
            elif name in self._kernel_explainers:
                # KernelExplainer is expensive; cap sample size for tractability.
                sample_positions = np.arange(len(X))
                if len(X) > 50:
                    sample_positions = np.random.default_rng(42).choice(
                        sample_positions, size=50, replace=False
                    )
                sample = X.iloc[sample_positions]
                shap_values = self._kernel_explainers[name].shap_values(
                    sample, nsamples=self.cfg.get("kernel_nsamples", 40)
                )
                if len(X) > 50:
                    padded = np.zeros((len(X), X.shape[1]))
                    padded[sample_positions] = shap_values
                    shap_values = padded
            else:
                continue
            scale = weight / max(total_model_weight, 1e-8)
            if self.ensemble.strategy == "weighted":
                _, std = score_statistics.get(name, (0.0, 1.0))
                scale /= max(std, 1e-8)
            combined += scale * np.asarray(shap_values)

        return pd.DataFrame(combined, columns=X.columns, index=X.index)

    def global_feature_importance(self, X: pd.DataFrame) -> pd.Series:
        """
        Global explanation: mean absolute SHAP value per feature across a
        sample of instances, indicating overall model reliance.
        """
        shap_df = self.explain_instances(X)
        return shap_df.abs().mean(axis=0).sort_values(ascending=False)

    def local_explanation(self, x_row: pd.DataFrame, top_k: Optional[int] = None) -> List[Dict]:
        """
        Local explanation for a single (user, product) row: the top-k
        features that most influenced this specific ranking score.

        Args:
            x_row: A single-row DataFrame (same columns as training data).
            top_k: Number of top contributing features to return; defaults
                to config value.

        Returns:
            List of {"feature": ..., "shap_value": ..., "direction": ...}
            dicts sorted by |shap_value| descending.
        """
        top_k = top_k or self.cfg["max_display_features"]
        shap_row = self.explain_instances(x_row).iloc[0]
        ranked = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)[:top_k]
        return [
            {
                "feature": feat,
                "shap_value": float(val),
                "direction": "increases_rank" if val > 0 else "decreases_rank",
            }
            for feat, val in ranked.items()
        ]

    def explain_pairwise_ranking(
        self, x_a: pd.DataFrame, x_b: pd.DataFrame, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        RankSHAP-style explanation: explain *why product A outranked
        product B* for a given user by attributing the difference in their
        SHAP contributions per feature.

        Args:
            x_a: Single-row feature vector for the higher-ranked product.
            x_b: Single-row feature vector for the lower-ranked product.
            top_k: Number of top differentiating features to return.

        Returns:
            List of {"feature", "delta_shap", "favors"} dicts, sorted by
            |delta_shap| descending -- the features that most explain the
            ranking gap between A and B.
        """
        top_k = top_k or self.cfg["max_display_features"]
        shap_a = self.explain_instances(x_a).iloc[0]
        shap_b = self.explain_instances(x_b).iloc[0]
        delta = shap_a - shap_b
        ranked = delta.reindex(delta.abs().sort_values(ascending=False).index)[:top_k]
        return [
            {
                "feature": feat,
                "delta_shap": float(val),
                "favors": "product_a" if val > 0 else "product_b",
            }
            for feat, val in ranked.items()
        ]
