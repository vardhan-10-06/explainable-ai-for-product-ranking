"""
tests/test_feature_engineering.py
====================================
Unit tests for the feature engineering pipeline.
"""
from feature_engineering.features import FeatureEngineer
from preprocessing.data_loader import DatasetLoader
from preprocessing.preprocess import Preprocessor
from ranking.train import _feature_columns


def test_full_feature_matrix_has_label_and_no_nans(small_synthetic_dataset):
    raw = DatasetLoader(dataset="amazon", raw_dir=small_synthetic_dataset).load()
    processed = Preprocessor().run(raw)
    fe = FeatureEngineer()
    matrix = fe.build_full_feature_matrix(
        processed.users, processed.products, processed.interactions, processed.reviews
    )
    assert "label" in matrix.columns
    assert matrix.isnull().sum().sum() == 0
    assert set(matrix["label"].unique()).issubset({0, 1})
    assert "pair_has_purchase" not in _feature_columns(matrix)
    assert "label" not in _feature_columns(matrix)


def test_candidate_features_include_unseen_catalog_products(small_synthetic_dataset):
    raw = DatasetLoader(dataset="amazon", raw_dir=small_synthetic_dataset).load()
    processed = Preprocessor().run(raw)
    fe = FeatureEngineer()
    matrix = fe.build_full_feature_matrix(
        processed.users, processed.products, processed.interactions, processed.reviews
    )

    user_id = matrix["user_id"].iloc[0]
    seen_products = set(matrix.loc[matrix["user_id"] == user_id, "product_id"])
    unseen_product = next(
        product_id for product_id in processed.products["product_id"]
        if product_id not in seen_products
    )

    candidates = fe.build_candidate_feature_matrix(user_id, [unseen_product])

    assert candidates["product_id"].tolist() == [unseen_product]
    assert candidates["pair_event_count"].iloc[0] == 0
    assert candidates["pair_has_purchase"].iloc[0] == 0
    assert "label" not in candidates.columns
