"""Normalize the UCSD Amazon Electronics 5-core dataset for this project.

The downloaded UCSD files are kept untouched under ``data/raw/ucsd_electronics``.
This importer writes the application's canonical four-table schema to a
separate directory so an import can be inspected before it becomes active.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # UCSD metadata is distributed as Python-literal records,
                    # while the review file uses strict JSON.
                    yield ast.literal_eval(line)


def _days_ago(unix_time: int, newest_time: int) -> int:
    return max(0, int((newest_time - int(unix_time)) / 86400))


def import_dataset(source_dir: Path, output_dir: Path) -> dict[str, int]:
    reviews_path = source_dir / "reviews_Electronics_5.json.gz"
    metadata_path = source_dir / "meta_Electronics.json.gz"
    if not reviews_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Expected both UCSD Electronics .json.gz files in the source directory.")

    review_rows: list[dict[str, Any]] = []
    product_stats: dict[str, list[float]] = defaultdict(list)
    reviewer_ids: set[str] = set()
    newest_time = 0

    for record in _read_json_lines(reviews_path):
        asin = str(record.get("asin", "")).strip()
        reviewer_id = str(record.get("reviewerID", "")).strip()
        if not asin or not reviewer_id:
            continue
        timestamp = int(record.get("unixReviewTime", 0) or 0)
        newest_time = max(newest_time, timestamp)
        rating = float(record.get("overall", 0) or 0)
        reviewer_ids.add(reviewer_id)
        product_stats[asin].append(rating)
        review_rows.append({
            "review_id": f"{reviewer_id}_{asin}_{timestamp}",
            "user_id": reviewer_id,
            "product_id": asin,
            "rating": rating,
            "text": str(record.get("reviewText", "") or ""),
            "days_ago": timestamp,
        })

    products: list[dict[str, Any]] = []
    product_ids = set(product_stats)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for record in _read_json_lines(metadata_path):
        asin = str(record.get("asin", "")).strip()
        if asin in product_ids:
            metadata_by_id[asin] = record

    for asin, ratings in product_stats.items():
        metadata = metadata_by_id.get(asin, {})
        categories = metadata.get("categories") or [["Electronics"]]
        category_path = categories[0] if categories and categories[0] else ["Electronics"]
        brand = str(metadata.get("brand") or "Unknown")
        price = metadata.get("price", 0.0)
        try:
            price = float(str(price).replace("$", "").replace(",", "").strip())
        except (TypeError, ValueError):
            price = 0.0
        products.append({
            "product_id": asin,
            "title": str(metadata.get("title") or asin),
            "category": str(category_path[-1] if category_path else "Electronics"),
            "category_path": " > ".join(str(item) for item in category_path),
            "brand": brand,
            "price": price,
            "avg_rating": sum(ratings) / len(ratings),
            "num_reviews": len(ratings),
            "launch_days_ago": 0,
        })

    users = pd.DataFrame({
        "user_id": sorted(reviewer_ids),
        "signup_days_ago": 0,
        "avg_session_len": 1.0,
    })
    reviews = pd.DataFrame(review_rows)
    interactions = reviews[["user_id", "product_id", "rating", "days_ago"]].copy()
    # Reviews are explicit user-item feedback rather than checkout logs.
    # Treat high ratings as positive purchase-like signals and low ratings as
    # negative views so ranking training has both classes.
    interactions["event_type"] = interactions["rating"].map(
        lambda rating: "purchase" if float(rating) >= 4.0 else "view"
    )
    interactions["days_ago"] = interactions["days_ago"].map(lambda value: _days_ago(value, newest_time))
    interactions = interactions[["user_id", "product_id", "event_type", "days_ago", "rating"]]
    if not reviews.empty:
        reviews["days_ago"] = reviews["days_ago"].map(lambda value: _days_ago(value, newest_time))

    output_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(output_dir / "users.csv", index=False)
    pd.DataFrame(products).to_csv(output_dir / "products.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    reviews.to_csv(output_dir / "reviews.csv", index=False)
    return {
        "users": len(users),
        "products": len(products),
        "interactions": len(interactions),
        "reviews": len(reviews),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import UCSD Amazon Electronics 5-core data")
    parser.add_argument("--source-dir", default="data/raw/ucsd_electronics")
    parser.add_argument("--output-dir", default="data/raw/ucsd_electronics_normalized")
    args = parser.parse_args()
    counts = import_dataset(Path(args.source_dir), Path(args.output_dir))
    print(json.dumps(counts, indent=2))
