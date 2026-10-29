"""
ranking/models.py
===================
Implements the individual ranking models used by the hybrid ensemble:
  * LightGBMRanker  - gradient-boosted LambdaRank
  * XGBoostRanker    - gradient-boosted pairwise ranking
  * NeuralRanker      - feed-forward network trained with a pairwise hinge loss
  * DeepFMRanker      - factorization-machine + deep network hybrid, capturing
                         both low-order (FM) and high-order (MLP) feature
                         interactions, which is why it complements the two
                         tree ensembles above (trees split axis-aligned;
                         DeepFM learns feature-crossing interactions).

Every model exposes the same three methods: fit(), predict(), and
feature_importance(), so ranking/ensemble.py can treat them uniformly.
"""
from __future__ import annotations

from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

from common.config import get_config, get_logger

logger = get_logger(__name__)


def _make_group_sizes(group_ids: pd.Series) -> List[int]:
    """LightGBM/XGBoost ranking objectives require group sizes for query groups (here: per-user)."""
    return group_ids.value_counts(sort=False).reindex(group_ids.unique()).tolist()


class LightGBMRanker:
    """Wraps LightGBM's LambdaRank objective for learning-to-rank."""

    def __init__(self):
        cfg = get_config()["ranking"]["models"]["lightgbm"]
        self.params = dict(
            objective=cfg["objective"],
            num_leaves=cfg["num_leaves"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            verbosity=-1,
        )
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series) -> "LightGBMRanker":
        """Fit the LambdaRank model. `group_ids` should be the user_id per row (query grouping)."""
        self.feature_names_ = list(X.columns)
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X, y, group=_make_group_sizes(group_ids))
        logger.info("LightGBMRanker trained on %d rows, %d features.", len(X), X.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return relevance scores (higher = more relevant)."""
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        """Return gain-based feature importances, indexed by feature name."""
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)


class XGBoostRanker:
    """Wraps XGBoost's pairwise ranking objective."""

    def __init__(self):
        cfg = get_config()["ranking"]["models"]["xgboost"]
        self.params = dict(
            objective=cfg["objective"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            eval_metric="ndcg",
        )
        self.model: Optional[xgb.XGBRanker] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series) -> "XGBoostRanker":
        """Fit the pairwise ranker. `group_ids` should be the user_id per row (query grouping)."""
        self.feature_names_ = list(X.columns)
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, y, group=_make_group_sizes(group_ids))
        logger.info("XGBoostRanker trained on %d rows, %d features.", len(X), X.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return relevance scores (higher = more relevant)."""
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        """Return gain-based feature importances, indexed by feature name."""
        booster = self.model.get_booster()
        raw = booster.get_score(importance_type="gain")
        # XGBoost names features f0, f1, ...; remap back to real column names.
        remapped = {self.feature_names_[int(k[1:])]: v for k, v in raw.items()}
        return pd.Series(remapped).reindex(self.feature_names_).fillna(0.0).sort_values(ascending=False)


class _FeedForwardNet(nn.Module):
    """Simple MLP scorer used by NeuralRanker."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralRanker:
    """
    Feed-forward neural ranking network trained with a pairwise hinge loss
    (RankNet-style): for each (positive, negative) pair from the same user,
    push score(positive) above score(negative) by a margin.
    """

    def __init__(self):
        cfg = get_config()["ranking"]["models"]["neural_ranker"]
        self.hidden_dims = cfg["hidden_dims"]
        self.dropout = cfg["dropout"]
        self.epochs = cfg["epochs"]
        self.batch_size = cfg["batch_size"]
        self.lr = cfg["lr"]
        self.model: Optional[_FeedForwardNet] = None
        self.feature_names_: List[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series) -> "NeuralRanker":
        """Fit the neural ranker using pairwise hinge loss over within-user pairs."""
        self.feature_names_ = list(X.columns)
        torch.manual_seed(get_config()["project"]["seed"])

        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)
        self.model = _FeedForwardNet(X.shape[1], self.hidden_dims, self.dropout).to(self._device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MarginRankingLoss(margin=1.0)

        pairs = self._build_pairs(y_t, group_ids)
        if not pairs:
            logger.warning("NeuralRanker: no valid ranking pairs found; skipping training.")
            return self

        self.model.train()
        for epoch in range(self.epochs):
            np.random.shuffle(pairs)
            total_loss = 0.0
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                idx_pos = torch.tensor([p[0] for p in batch])
                idx_neg = torch.tensor([p[1] for p in batch])
                optimizer.zero_grad()
                score_pos = self.model(X_t[idx_pos].to(self._device))
                score_neg = self.model(X_t[idx_neg].to(self._device))
                target = torch.ones_like(score_pos)
                loss = loss_fn(score_pos, score_neg, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("NeuralRanker epoch %d/%d - loss=%.4f", epoch + 1, self.epochs, total_loss)
        return self

    def _build_pairs(self, y: torch.Tensor, group_ids: pd.Series) -> List[tuple]:
        pairs = []
        y_np = y.numpy()
        rng = np.random.default_rng(get_config()["project"]["seed"])
        for _, idx in group_ids.groupby(group_ids).groups.items():
            idx = list(idx)
            pos = [i for i in idx if y_np[i] > 0]
            neg = [i for i in idx if y_np[i] == 0]
            pair_count = len(pos) * len(neg)
            pair_limit = max(1_000, len(idx) * 10)
            if pair_count <= pair_limit:
                pairs.extend((p, n) for p in pos for n in neg)
            elif pair_count:
                sampled_count = min(pair_count, pair_limit)
                sampled_pos = rng.choice(pos, size=sampled_count, replace=True)
                sampled_neg = rng.choice(neg, size=sampled_count, replace=True)
                pairs.extend(zip(sampled_pos.tolist(), sampled_neg.tolist()))
        return pairs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return relevance scores (higher = more relevant)."""
        if self.model is None:
            return np.zeros(len(X))
        self.model.eval()
        with torch.no_grad():
            values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            X_t = torch.tensor(values, dtype=torch.float32).to(self._device)
            return self.model(X_t).cpu().numpy()

    def feature_importance(self) -> pd.Series:
        """
        Approximate feature importance via the magnitude of first-layer
        weights (a common lightweight proxy for MLP feature relevance;
        SHAP's KernelExplainer, used in explainability/, gives the exact
        per-prediction attributions instead).
        """
        first_layer = next(m for m in self.model.net if isinstance(m, nn.Linear))
        importances = first_layer.weight.detach().abs().mean(dim=0).cpu().numpy()
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)


class DeepFMRanker:
    """
    Simplified DeepFM: combines a factorization-machine (FM) component that
    models pairwise feature interactions in closed form with a deep MLP
    component over the same embedded features, then sums both outputs.
    Chosen over a plain MLP because FM explicitly captures second-order
    feature crosses (e.g. category x price) that trees can only approximate
    via many splits and that a plain MLP must learn implicitly.
    """

    def __init__(self, embedding_dim: int = 8):
        cfg = get_config()["ranking"]["models"]["neural_ranker"]
        self.embedding_dim = embedding_dim
        self.hidden_dims = cfg["hidden_dims"]
        self.epochs = cfg["epochs"]
        self.batch_size = cfg["batch_size"]
        self.lr = cfg["lr"]
        self.model: Optional[nn.Module] = None
        self.feature_names_: List[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _DeepFMModule(nn.Module):
        def __init__(self, num_features: int, embedding_dim: int, hidden_dims: List[int]):
            super().__init__()
            self.linear = nn.Linear(num_features, 1)
            self.embeddings = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.01)
            mlp_layers = []
            prev = num_features * embedding_dim
            for h in hidden_dims:
                mlp_layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            mlp_layers.append(nn.Linear(prev, 1))
            self.mlp = nn.Sequential(*mlp_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            linear_term = self.linear(x).squeeze(-1)
            # Second-order FM interaction term: 0.5 * sum((sum vx)^2 - sum(vx)^2)
            vx = x.unsqueeze(-1) * self.embeddings.unsqueeze(0)  # (batch, features, emb_dim)
            sum_square = vx.sum(dim=1) ** 2
            square_sum = (vx ** 2).sum(dim=1)
            fm_term = 0.5 * (sum_square - square_sum).sum(dim=1)
            deep_term = self.mlp(vx.reshape(x.shape[0], -1)).squeeze(-1)
            return linear_term + fm_term + deep_term

    def fit(self, X: pd.DataFrame, y: pd.Series, group_ids: pd.Series) -> "DeepFMRanker":
        """Fit DeepFM with pairwise hinge loss, identical training regime to NeuralRanker."""
        self.feature_names_ = list(X.columns)
        torch.manual_seed(get_config()["project"]["seed"])
        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)

        self.model = self._DeepFMModule(X.shape[1], self.embedding_dim, self.hidden_dims).to(self._device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MarginRankingLoss(margin=1.0)

        pairs = NeuralRanker()._build_pairs(y_t, group_ids)  # reuse pair-construction logic
        if not pairs:
            logger.warning("DeepFMRanker: no valid ranking pairs found; skipping training.")
            return self

        self.model.train()
        for epoch in range(self.epochs):
            np.random.shuffle(pairs)
            total_loss = 0.0
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                idx_pos = torch.tensor([p[0] for p in batch])
                idx_neg = torch.tensor([p[1] for p in batch])
                optimizer.zero_grad()
                score_pos = self.model(X_t[idx_pos].to(self._device))
                score_neg = self.model(X_t[idx_neg].to(self._device))
                target = torch.ones_like(score_pos)
                loss = loss_fn(score_pos, score_neg, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("DeepFMRanker epoch %d/%d - loss=%.4f", epoch + 1, self.epochs, total_loss)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return relevance scores (higher = more relevant)."""
        if self.model is None:
            return np.zeros(len(X))
        self.model.eval()
        with torch.no_grad():
            values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            X_t = torch.tensor(values, dtype=torch.float32).to(self._device)
            return self.model(X_t).cpu().numpy()

    def feature_importance(self) -> pd.Series:
        """Approximate importance via the linear component's absolute weights."""
        importances = self.model.linear.weight.detach().abs().squeeze(0).cpu().numpy()
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)
