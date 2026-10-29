"""
tests/test_data_pipeline.py
=============================
Unit tests for dataset generation, loading, and preprocessing.
"""
import pandas as pd

from feature_engineering.features import FeatureEngineer
from preprocessing.data_loader import DatasetLoader
from preprocessing.preprocess import Preprocessor, split_train_val_test
from ranking.train import _feature_columns
from evaluation import evaluate_datasets
from evaluation.evaluate import _category_metrics
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from backend.service import RankingService


def test_generated_dataset_has_expected_columns(small_synthetic_dataset):
    loader = DatasetLoader(dataset="amazon", raw_dir=small_synthetic_dataset)
    tables = loader.load()
    assert "user_id" in tables.users.columns
    assert "product_id" in tables.products.columns
    assert {"user_id", "product_id", "event_type", "days_ago"}.issubset(tables.interactions.columns)


def test_preprocessing_produces_encoded_and_scaled_columns(small_synthetic_dataset):
    loader = DatasetLoader(dataset="amazon", raw_dir=small_synthetic_dataset)
    raw = loader.load()
    processed = Preprocessor().run(raw)
    assert "category_encoded" in processed.products.columns
    assert "price_scaled" in processed.products.columns
    assert processed.products.isnull().sum().sum() == 0


def test_train_val_test_split_sizes_sum_correctly(small_synthetic_dataset):
    loader = DatasetLoader(dataset="amazon", raw_dir=small_synthetic_dataset)
    raw = loader.load()
    processed = Preprocessor().run(raw)
    train, val, test = split_train_val_test(processed.interactions)
    assert len(train) + len(val) + len(test) == len(processed.interactions)
    assert len(train) > 0


def test_flipkart_loader_uses_category_groups_and_rating_labels(tmp_path):
    source = pd.DataFrame({
        "Category": ["mobiles"] * 4 + ["laptops"] * 4,
        "Title": [f"Product {i}" for i in range(8)],
        "Product Link": [f"https://shop.test/item?pid={i}" for i in range(8)],
        "Rating": [4.5, 3.0, 4.0, None, 2.5, 4.2, 3.5, 4.8],
        "Price": ["₹1,000"] * 8,
        "Ratings & Reviews": ["1,234"] * 8,
        "Key Features": [f"Feature specification {i}" for i in range(8)],
        "Timestamp": ["2025-07-21 12:00:00"] * 8,
    })
    source.loc[len(source)] = [
        "mobiles", "Product 0", "https://shop.test/item?pid=0", 3.5,
        "₹1,200", "1,234", "Updated specification", "2025-07-21 12:00:00",
    ]
    source.to_csv(tmp_path / "flipkart_combined.csv", index=False)

    raw = DatasetLoader(dataset="flipkart", raw_dir=str(tmp_path)).load()
    assert set(raw.users["user_id"]) == {"mobiles", "laptops"}
    assert len(raw.interactions) == 7
    assert raw.interactions["event_type"].eq("view").all()
    assert raw.interactions["relevance"].sum() == 4
    assert raw.interactions.loc[
        raw.interactions["product_id"] == "0", "rating"
    ].item() == 4.0

    processed = Preprocessor().run(raw)
    matrix = FeatureEngineer().build_full_feature_matrix(
        processed.users, processed.products, processed.interactions, processed.reviews
    )
    assert matrix["label"].sum() == 4
    assert "rating" not in matrix.columns
    assert matrix["pair_has_purchase"].eq(0).all()
    assert processed.products["avg_rating_scaled"].eq(0).all()
    feature_columns = _feature_columns(matrix)
    assert "price_scaled" not in feature_columns
    assert "avg_rating_scaled" not in feature_columns
    assert "num_reviews_scaled" in feature_columns
    assert any(column.startswith("product_text_") for column in feature_columns)


def test_dataset_discovery_reports_required_files_and_physical_csv_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_datasets, "PROJECT_ROOT", tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    flipkart_file = raw_dir / "flipkart_combined.csv"
    flipkart_file.write_text('Title\n"line one\nline two"\n', encoding="utf-8")
    for table_name in evaluate_datasets.STANDARD_TABLES:
        (raw_dir / table_name).write_text("id\n1\n", encoding="utf-8")

    entries = {
        entry["dataset_id"]: entry
        for entry in evaluate_datasets.discover_datasets()
    }
    assert entries["flipkart"]["status"] == "ready"
    assert entries["flipkart"]["csv_data_lines"]["flipkart_combined.csv"] == 2
    assert entries["amazon_sample"]["status"] == "ready"
    assert entries["amazon_electronics_runtime"]["status"] == "not_present"
    assert entries["retailrocket"]["status"] == "not_present"


def test_dataset_report_converts_non_finite_metrics_to_json_null():
    assert evaluate_datasets._json_safe({
        "metrics": {"roc_auc": float("nan"), "ndcg": 0.75}
    }) == {"metrics": {"roc_auc": None, "ndcg": 0.75}}


def test_graph_embedding_factorization_accepts_sparse_cooccurrence():
    builder = KnowledgeGraphBuilder()
    matrix = builder._build_cooccurrence(
        [["Product:a", "Product:b", "Category:phones"]],
        ["Product:a", "Product:b"],
    )

    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == matrix[1, 0] == 1
    assert builder._factorize(matrix, dim=4).shape == (2, 4)


def test_evaluation_metrics_are_separated_by_selected_category():
    test_rows = pd.DataFrame({
        "user_id": ["mobiles", "mobiles", "laptops", "laptops"],
        "product_id": ["m1", "m2", "l1", "l2"],
        "label": [1, 0, 0, 1],
        "ensemble_score": [0.9, 0.1, 0.1, 0.9],
    })

    metrics = _category_metrics(test_rows, [10], score_col="ensemble_score")

    assert set(metrics) == {"mobiles", "laptops"}
    assert metrics["mobiles"]["test_rows"] == 2
    assert metrics["laptops"]["products"] == 2
    assert metrics["mobiles"]["metrics"]["ndcg_at_10"] == 1.0
    assert metrics["laptops"]["metrics"]["ndcg_at_10"] == 1.0


def test_catalog_search_is_free_text_with_selected_category_filter():
    service = RankingService()
    service.ensure_loaded = lambda: None
    service.products = pd.DataFrame({
        "product_id": ["earphone-1", "laptop-1", "earphone-2"],
        "title": ["Wireless headset", "Gaming laptop", "C++ Audio Buds"],
        "category": ["earphones", "laptops", "earphones"],
        "brand": ["AudioCo", "ComputeCo", "Sound+"],
    })

    results = service.search_products("C++", " EARPHONES ", 10)

    assert [row["product_id"] for row in results] == ["earphone-2"]
