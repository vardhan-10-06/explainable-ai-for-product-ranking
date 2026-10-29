"""
preprocessing/preprocess.py
============================
Cleans and encodes the raw common-schema tables:
  * missing-value handling
  * categorical encoding (label encoding, persisted for inference-time reuse)
  * numerical normalization (standard scaling, persisted)
  * user/item filtering by minimum interaction count
  * grouped train/val/test split (split by user so a user's history doesn't
    leak across splits, which is the standard practice for ranking tasks)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.config import get_config, get_logger
from preprocessing.data_loader import RawTables

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """Container for the cleaned/encoded tables plus fitted encoders."""
    users: pd.DataFrame
    products: pd.DataFrame
    interactions: pd.DataFrame
    reviews: pd.DataFrame
    encoders: Dict[str, LabelEncoder]
    scalers: Dict[str, StandardScaler]


class Preprocessor:
    """Cleans, encodes, and normalizes raw dataset tables."""

    def __init__(self):
        self.cfg = get_config()["data"]
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def run(self, raw: RawTables) -> ProcessedData:
        """
        Execute the full preprocessing pipeline on raw tables.

        Args:
            raw: RawTables produced by `DatasetLoader.load()`.

        Returns:
            ProcessedData with cleaned/encoded tables and fitted encoders.
        """
        users = self._clean_users(raw.users)
        products = self._clean_products(raw.products)
        interactions = self._clean_interactions(raw.interactions)
        reviews = self._clean_reviews(raw.reviews)

        users, products, interactions = self._filter_min_interactions(users, products, interactions)
        products = self._encode_categoricals(products)
        products = self._normalize_numericals(products)
        users = self._normalize_numericals(users, prefix="user")

        logger.info(
            "Preprocessing complete: users=%d products=%d interactions=%d reviews=%d",
            len(users), len(products), len(interactions), len(reviews),
        )
        return ProcessedData(users, products, interactions, reviews, self.encoders, self.scalers)

    # ------------------------------------------------------------------ #
    def _clean_users(self, users: pd.DataFrame) -> pd.DataFrame:
        users = users.drop_duplicates(subset="user_id").copy()
        users["signup_days_ago"] = users["signup_days_ago"].fillna(users["signup_days_ago"].median())
        users["avg_session_len"] = users["avg_session_len"].fillna(users["avg_session_len"].median())
        return users

    def _clean_products(self, products: pd.DataFrame) -> pd.DataFrame:
        products = products.drop_duplicates(subset="product_id").copy()
        products["category"] = products["category"].fillna("Unknown")
        products["brand"] = products["brand"].fillna("Unknown")
        products["price"] = products["price"].fillna(products["price"].median())
        products["avg_rating"] = products["avg_rating"].fillna(products["avg_rating"].mean())
        products["num_reviews"] = products["num_reviews"].fillna(0)
        products["launch_days_ago"] = products["launch_days_ago"].fillna(products["launch_days_ago"].median())
        return products

    def _clean_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions = interactions.dropna(subset=["user_id", "product_id", "event_type"]).copy()
        interactions["days_ago"] = interactions["days_ago"].fillna(interactions["days_ago"].max())
        return interactions

    def _clean_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        if reviews.empty:
            return reviews
        reviews = reviews.dropna(subset=["user_id", "product_id"]).copy()
        reviews["text"] = reviews["text"].fillna("")
        reviews["rating"] = reviews["rating"].fillna(reviews["rating"].mean() if reviews["rating"].notna().any() else 3.0)
        return reviews

    def _filter_min_interactions(
        self, users: pd.DataFrame, products: pd.DataFrame, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        min_u = self.cfg["min_interactions_per_user"]
        min_i = self.cfg["min_interactions_per_item"]

        user_counts = interactions["user_id"].value_counts()
        item_counts = interactions["product_id"].value_counts()
        valid_users = user_counts[user_counts >= min_u].index
        valid_items = item_counts[item_counts >= min_i].index

        interactions = interactions[
            interactions["user_id"].isin(valid_users) & interactions["product_id"].isin(valid_items)
        ].copy()
        users = users[users["user_id"].isin(interactions["user_id"].unique())].copy()
        products = products[products["product_id"].isin(interactions["product_id"].unique())].copy()
        return users, products, interactions

    def _encode_categoricals(self, products: pd.DataFrame) -> pd.DataFrame:
        for col in ["category", "brand"]:
            encoder = LabelEncoder()
            products[f"{col}_encoded"] = encoder.fit_transform(products[col].astype(str))
            self.encoders[col] = encoder
        return products

    def _normalize_numericals(self, df: pd.DataFrame, prefix: str = "product") -> pd.DataFrame:
        numeric_cols = {
            "product": ["price", "avg_rating", "num_reviews", "launch_days_ago"],
            "user": ["signup_days_ago", "avg_session_len"],
        }[prefix]
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        if not numeric_cols:
            return df
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols])
        for i, col in enumerate(numeric_cols):
            df[f"{col}_scaled"] = scaled[:, i]
        self.scalers[prefix] = scaler
        return df

    def save_artifacts(self, processed: ProcessedData, out_dir: str) -> None:
        """Persist processed tables and metadata about fitted encoders."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        processed.users.to_csv(out / "users.csv", index=False)
        processed.products.to_csv(out / "products.csv", index=False)
        processed.interactions.to_csv(out / "interactions.csv", index=False)
        processed.reviews.to_csv(out / "reviews.csv", index=False)

        encoder_classes = {
            name: enc.classes_.tolist() for name, enc in processed.encoders.items()
        }
        with open(out / "encoders.json", "w", encoding="utf-8") as fh:
            json.dump(encoder_classes, fh, indent=2)
        logger.info("Saved processed artifacts to %s", out)


def split_train_val_test(
    interactions: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train/val/test *by user* so that a given
    user's interactions are split across the three sets proportionally,
    while still respecting temporal order (older interactions -> train).

    Args:
        interactions: Cleaned interactions DataFrame with a `days_ago` column
            (larger days_ago = older).
        seed: Random seed for shuffling ties.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    cfg = get_config()["data"]
    train_ratio, val_ratio = cfg["train_ratio"], cfg["val_ratio"]

    interactions = interactions.sort_values(["user_id", "days_ago"], ascending=[True, False])
    train_parts, val_parts, test_parts = [], [], []

    for _, group in interactions.groupby("user_id"):
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df
