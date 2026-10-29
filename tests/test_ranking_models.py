"""
tests/test_ranking_models.py
==============================
Unit tests for individual ranking models and the hybrid ensemble.
"""
import pandas as pd

from ranking.ensemble import HybridRankingEnsemble
from ranking.models import LightGBMRanker


def _toy_data():
    X = pd.DataFrame({
        "f1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85],
        "f2": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    groups = pd.Series(["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"])
    return X, y, groups


def test_lightgbm_ranker_fits_and_predicts():
    X, y, groups = _toy_data()
    model = LightGBMRanker().fit(X, y, groups)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_hybrid_ensemble_weighted_strategy_runs_end_to_end():
    X, y, groups = _toy_data()
    ensemble = HybridRankingEnsemble()
    ensemble.strategy = "weighted"
    ensemble.fit(X, y, groups)
    scores = ensemble.predict(X)
    assert len(scores) == len(X)
    assert not any(pd.isna(scores))
    assert ensemble.predict(X.iloc[[1]])[0] == scores[1]
