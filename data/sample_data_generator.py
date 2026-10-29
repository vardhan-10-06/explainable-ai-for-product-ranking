"""
data/sample_data_generator.py
==============================
Generates a realistic *synthetic* dataset that mimics the schema of the
supported real-world datasets (Amazon Reviews, RetailRocket, MovieLens,
Yelp) so the whole pipeline can be run end-to-end without needing to
download multi-gigabyte source files first.

The generated schema is dataset-agnostic:
    users.csv         : user_id, signup_days_ago, avg_session_len
    products.csv       : product_id, category, brand, price, avg_rating,
                          num_reviews, launch_days_ago
    interactions.csv   : user_id, product_id, event_type, timestamp,
                          rating (nullable)
    reviews.csv         : review_id, user_id, product_id, rating, text,
                          timestamp

Real dataset loaders (see preprocessing/data_loader.py) read this exact
schema, so swapping in real Amazon/RetailRocket/MovieLens/Yelp dumps only
requires mapping their native columns onto this schema -- the rest of the
pipeline (feature engineering, KG, ranking, explainability) needs no
changes. This is the key design decision that keeps "support 4 datasets"
tractable: normalize at the loader boundary, not throughout the codebase.
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_logger

logger = get_logger(__name__)

CATEGORIES = [
    "Electronics", "Home & Kitchen", "Books", "Sports & Outdoors",
    "Beauty", "Toys & Games", "Clothing", "Grocery",
]
BRANDS = [f"Brand_{c}" for c in string.ascii_uppercase[:20]]
EVENT_TYPES = ["view", "cart", "purchase", "wishlist"]
EVENT_WEIGHTS = [0.60, 0.20, 0.15, 0.05]

REVIEW_SNIPPETS_POS = [
    "Works exactly as described and arrived quickly.",
    "Great value for the price, would buy again.",
    "Build quality exceeded my expectations.",
    "Exactly what I needed, easy to set up.",
]
REVIEW_SNIPPETS_NEG = [
    "Stopped working after a few weeks.",
    "Not as described, quite disappointed.",
    "Overpriced for what you actually get.",
    "Packaging was damaged on arrival.",
]
REVIEW_SNIPPETS_NEU = [
    "It's okay, does the job but nothing special.",
    "Average product, matches similar items in this price range.",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_users(n_users: int, seed: int) -> pd.DataFrame:
    """Generate synthetic user profiles."""
    rng = _rng(seed)
    return pd.DataFrame({
        "user_id": [f"U{idx:06d}" for idx in range(n_users)],
        "signup_days_ago": rng.integers(1, 1500, size=n_users),
        "avg_session_len": np.round(rng.gamma(2.0, 5.0, size=n_users), 2),
    })


def generate_products(n_products: int, seed: int) -> pd.DataFrame:
    """Generate synthetic product catalog with category/brand metadata."""
    rng = _rng(seed + 1)
    categories = rng.choice(CATEGORIES, size=n_products)
    brands = rng.choice(BRANDS, size=n_products)
    prices = np.round(rng.lognormal(mean=3.2, sigma=0.6, size=n_products), 2)
    ratings = np.clip(rng.normal(4.0, 0.6, size=n_products), 1.0, 5.0).round(2)
    num_reviews = rng.poisson(40, size=n_products)
    launch_days_ago = rng.integers(1, 2000, size=n_products)
    return pd.DataFrame({
        "product_id": [f"P{idx:06d}" for idx in range(n_products)],
        "category": categories,
        "brand": brands,
        "price": prices,
        "avg_rating": ratings,
        "num_reviews": num_reviews,
        "launch_days_ago": launch_days_ago,
    })


def generate_interactions(
    users: pd.DataFrame, products: pd.DataFrame, n_interactions: int, seed: int
) -> pd.DataFrame:
    """
    Generate synthetic user-product interactions with a mild popularity and
    category-affinity bias so downstream ranking models have real signal to
    learn from (pure randomness would make ranking meaningless to evaluate).
    """
    rng = _rng(seed + 2)
    user_ids = users["user_id"].values
    n_users = len(user_ids)

    # Give each user a latent preferred category to create learnable signal.
    user_pref_category = rng.choice(CATEGORIES, size=n_users)
    user_pref_map = dict(zip(user_ids, user_pref_category))

    product_popularity = products["num_reviews"].values.astype(float)
    product_popularity = product_popularity / product_popularity.sum()

    rows = []
    chosen_users = rng.choice(user_ids, size=n_interactions)
    for uid in chosen_users:
        pref_cat = user_pref_map[uid]
        same_cat_mask = (products["category"] == pref_cat).values
        if same_cat_mask.sum() > 0 and rng.random() < 0.65:
            candidate_idx = np.where(same_cat_mask)[0]
            weights = product_popularity[candidate_idx]
            weights = weights / weights.sum()
            pid_idx = rng.choice(candidate_idx, p=weights)
        else:
            pid_idx = rng.choice(len(products), p=product_popularity)

        event = rng.choice(EVENT_TYPES, p=EVENT_WEIGHTS)
        rating = None
        if event == "purchase" and rng.random() < 0.7:
            rating = int(np.clip(rng.normal(4.2, 0.9), 1, 5))
        rows.append((
            uid,
            products.iloc[pid_idx]["product_id"],
            event,
            int(rng.integers(1, 365)),  # timestamp as "days ago", simpler than epoch
            rating,
        ))

    df = pd.DataFrame(rows, columns=["user_id", "product_id", "event_type", "days_ago", "rating"])
    df = df.sort_values("days_ago", ascending=False).reset_index(drop=True)
    return df


def generate_reviews(interactions: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate free-text reviews for a subset of purchase interactions."""
    rng = _rng(seed + 3)
    purchases = interactions[interactions["event_type"] == "purchase"].copy()
    purchases = purchases[purchases["rating"].notna()]

    reviews = []
    for i, row in enumerate(purchases.itertuples(index=False)):
        if rng.random() > 0.5:  # not every purchase gets a written review
            continue
        rating = row.rating
        if rating >= 4:
            text = rng.choice(REVIEW_SNIPPETS_POS)
        elif rating <= 2:
            text = rng.choice(REVIEW_SNIPPETS_NEG)
        else:
            text = rng.choice(REVIEW_SNIPPETS_NEU)
        reviews.append((f"R{i:06d}", row.user_id, row.product_id, rating, text, row.days_ago))

    return pd.DataFrame(
        reviews, columns=["review_id", "user_id", "product_id", "rating", "text", "days_ago"]
    )


def generate_dataset(
    output_dir: str = "data/raw",
    n_users: int = 2000,
    n_products: int = 500,
    n_interactions: int = 40000,
    seed: int = 42,
) -> None:
    """
    Generate the full synthetic dataset (users, products, interactions,
    reviews) and write each table to CSV under `output_dir`.

    Args:
        output_dir: Directory to write the CSV files to.
        n_users: Number of synthetic users.
        n_products: Number of synthetic products.
        n_interactions: Number of user-product interaction events.
        seed: Random seed for full reproducibility.
    """
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d users...", n_users)
    users = generate_users(n_users, seed)
    logger.info("Generating %d products...", n_products)
    products = generate_products(n_products, seed)
    logger.info("Generating %d interactions...", n_interactions)
    interactions = generate_interactions(users, products, n_interactions, seed)
    logger.info("Generating reviews from purchase interactions...")
    reviews = generate_reviews(interactions, seed)

    users.to_csv(out / "users.csv", index=False)
    products.to_csv(out / "products.csv", index=False)
    interactions.to_csv(out / "interactions.csv", index=False)
    reviews.to_csv(out / "reviews.csv", index=False)

    logger.info(
        "Wrote users=%d products=%d interactions=%d reviews=%d to %s",
        len(users), len(products), len(interactions), len(reviews), out,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic product-ranking dataset")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--n-products", type=int, default=500)
    parser.add_argument("--n-interactions", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_products=args.n_products,
        n_interactions=args.n_interactions,
        seed=args.seed,
    )
