"""Create a responsive runtime slice from the complete UCSD import.

The complete normalized import remains available for research. The serving
and training pipeline uses this deterministic slice so the local application
does not require loading 1.7M interactions on every startup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare(source_dir: Path, output_dir: Path, max_users: int = 5000) -> dict[str, int]:
    users = pd.read_csv(source_dir / "users.csv", usecols=["user_id"]).head(max_users)
    selected_users = set(users["user_id"].astype(str))
    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_parts = []
    for chunk in pd.read_csv(source_dir / "interactions.csv", chunksize=100_000):
        part = chunk[chunk["user_id"].astype(str).isin(selected_users)]
        if not part.empty:
            interaction_parts.append(part)
    interactions = pd.concat(interaction_parts, ignore_index=True)
    product_ids = set(interactions["product_id"].astype(str))

    products = pd.read_csv(source_dir / "products.csv")
    products = products[products["product_id"].astype(str).isin(product_ids)]

    review_parts = []
    for chunk in pd.read_csv(source_dir / "reviews.csv", chunksize=50_000):
        part = chunk[chunk["user_id"].astype(str).isin(selected_users)]
        if not part.empty:
            review_parts.append(part)
    reviews = pd.concat(review_parts, ignore_index=True) if review_parts else pd.DataFrame()

    users = pd.read_csv(source_dir / "users.csv")
    users = users[users["user_id"].astype(str).isin(selected_users)]
    users.to_csv(output_dir / "users.csv", index=False)
    products.to_csv(output_dir / "products.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    reviews.to_csv(output_dir / "reviews.csv", index=False)
    return {
        "users": len(users),
        "products": len(products),
        "interactions": len(interactions),
        "reviews": len(reviews),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a responsive UCSD runtime dataset")
    parser.add_argument("--source-dir", default="data/raw/ucsd_electronics_normalized")
    parser.add_argument("--output-dir", default="data/raw/ucsd_electronics_runtime")
    parser.add_argument("--max-users", type=int, default=5000)
    args = parser.parse_args()
    print(prepare(Path(args.source_dir), Path(args.output_dir), args.max_users))
