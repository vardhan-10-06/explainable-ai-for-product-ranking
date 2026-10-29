"""Train and evaluate each locally available dataset as an independent run."""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from common.config import PROJECT_ROOT, get_config, get_logger
from evaluation.evaluate import evaluate_model
from ranking.train import train_and_save

logger = get_logger(__name__)

MAX_LOCAL_EVALUATION_ROWS = 100_000
STANDARD_TABLES = ("users.csv", "products.csv", "interactions.csv", "reviews.csv")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_data_lines(path: Path) -> int:
    """Count physical CSV data lines without loading the table into memory."""
    line_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            line_count += chunk.count(b"\n")
    return max(line_count - 1, 0)


def _dataset_entry(
    dataset_id: str,
    name: str,
    loader_dataset: str,
    raw_dir: Path,
    files: List[Path],
    evaluate: bool,
    note: str = "",
) -> Dict:
    present_files = [path for path in files if path.exists()]
    size_bytes = sum(path.stat().st_size for path in present_files)
    data_lines = {
        path.name: _csv_data_lines(path)
        for path in present_files
        if path.suffix.lower() == ".csv"
    }
    return {
        "dataset_id": dataset_id,
        "name": name,
        "loader_dataset": loader_dataset,
        "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
        "status": "ready" if evaluate else "available_not_evaluated",
        "raw_size_bytes": size_bytes,
        "raw_size_mb": round(size_bytes / (1024 * 1024), 2),
        "source_files": [str(path.relative_to(PROJECT_ROOT)) for path in present_files],
        "csv_data_lines": data_lines,
        "note": note,
    }


def discover_datasets() -> List[Dict]:
    """Describe project datasets on disk and which ones fit a local run."""
    raw_root = PROJECT_ROOT / "data" / "raw"
    candidates = [
        (
            "flipkart",
            "Flipkart Combined",
            "flipkart",
            raw_root,
            [raw_root / "flipkart_combined.csv"],
            True,
            "Ratings of 4+ are the relevance label; price is display-only.",
        ),
        (
            "amazon_sample",
            "Amazon-format sample",
            "amazon",
            raw_root,
            [raw_root / filename for filename in STANDARD_TABLES],
            True,
            "The small common-schema sample currently stored at data/raw.",
        ),
        (
            "amazon_electronics_runtime",
            "Amazon Electronics runtime slice",
            "amazon",
            raw_root / "ucsd_electronics_runtime",
            [raw_root / "ucsd_electronics_runtime" / filename for filename in STANDARD_TABLES],
            True,
            "5,000-user runtime slice of the normalized UCSD Electronics data.",
        ),
        (
            "amazon_electronics_full",
            "Amazon Electronics full normalized",
            "amazon",
            raw_root / "ucsd_electronics_normalized",
            [raw_root / "ucsd_electronics_normalized" / filename for filename in STANDARD_TABLES],
            False,
            "Full 1.69M-interaction data is inventoried but not trained locally; use the runtime slice for comparable local metrics.",
        ),
        (
            "retailrocket",
            "RetailRocket",
            "retailrocket",
            raw_root / "retailrocket",
            [raw_root / "retailrocket" / "events.csv"],
            False,
            "Supply the official events.csv export to evaluate this dataset.",
        ),
        (
            "movielens",
            "MovieLens",
            "movielens",
            raw_root / "movielens",
            [raw_root / "movielens" / "ratings.csv", raw_root / "movielens" / "movies.csv"],
            False,
            "Supply ratings.csv and movies.csv to evaluate this dataset.",
        ),
        (
            "yelp",
            "Yelp",
            "yelp",
            raw_root / "yelp",
            [
                raw_root / "yelp" / "yelp_academic_dataset_review.json",
                raw_root / "yelp" / "yelp_academic_dataset_business.json",
            ],
            False,
            "Supply the Yelp review and business JSONL files to evaluate this dataset.",
        ),
    ]

    entries = []
    for dataset_id, name, loader, raw_dir, files, can_evaluate, note in candidates:
        present = [path for path in files if path.is_file()]
        if loader == "amazon" and dataset_id == "amazon_sample":
            ready = len(present) == len(STANDARD_TABLES)
        elif dataset_id == "amazon_electronics_runtime":
            ready = len(present) == len(STANDARD_TABLES)
        elif dataset_id == "amazon_electronics_full":
            ready = len(present) == len(STANDARD_TABLES)
        elif dataset_id == "movielens":
            ready = len(present) == 2
        elif dataset_id == "yelp":
            ready = len(present) == 2
        else:
            ready = len(present) == len(files)

        entry = _dataset_entry(
            dataset_id, name, loader, raw_dir, present,
            evaluate=can_evaluate and ready, note=note,
        )
        if not ready:
            entry["status"] = "not_present"
            entry["note"] = note
        elif not can_evaluate:
            entry["status"] = "available_not_evaluated"
        entries.append(entry)
    return entries


def evaluate_available_datasets(output_path: str = "results/dataset_comparison.json") -> Dict:
    """Train, evaluate, and compare every present dataset that fits local resources."""
    config = get_config()
    data_config = config["data"]
    original_data_config = dict(data_config)
    datasets = discover_datasets()

    try:
        for dataset in datasets:
            if dataset["status"] != "ready":
                continue

            data_config["dataset"] = dataset["loader_dataset"]
            data_config["raw_dir"] = dataset["raw_dir"]
            run_dir = PROJECT_ROOT / "results" / "datasets" / dataset["dataset_id"]
            artifact_path = run_dir / "hybrid_ensemble.pkl"
            run_dir.mkdir(parents=True, exist_ok=True)

            train_start = time.perf_counter()
            training = train_and_save(str(artifact_path))
            dataset["training_seconds"] = round(time.perf_counter() - train_start, 3)
            dataset["split"] = training
            dataset["status"] = "evaluating"

            evaluation_start = time.perf_counter()
            report = evaluate_model(
                ensemble_path=str(artifact_path),
                output_dir=str(run_dir),
            )
            dataset["evaluation_seconds"] = round(time.perf_counter() - evaluation_start, 3)
            dataset["metrics"] = report["hybrid_ensemble"]
            dataset["baseline_metrics"] = report["popularity_baseline"]
            dataset["efficiency"] = report["efficiency"]
            dataset["category_metrics"] = report.get("category_metrics", {})
            dataset["category_baseline_metrics"] = report.get(
                "category_baseline_metrics", {}
            )
            dataset["category_brand_metrics"] = report.get(
                "category_brand_metrics", {}
            )
            dataset["category_brand_baseline_metrics"] = report.get(
                "category_brand_baseline_metrics", {}
            )
            dataset["status"] = "evaluated"
            dataset["artifact"] = str(artifact_path.relative_to(PROJECT_ROOT))
            logger.info(
                "Evaluated %s: rows=%d NDCG@10=%.4f",
                dataset["name"],
                training["train_rows"] + training["validation_rows"] + training["test_rows"],
                report["hybrid_ensemble"].get("ndcg_at_10", 0.0),
            )
    finally:
        data_config.clear()
        data_config.update(original_data_config)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count_note": (
            "CSV size is file size on disk. CSV data-line counts are physical newline counts; "
            "quoted embedded newlines in text can make those counts slightly higher than records."
        ),
        "datasets": datasets,
        "metric_definitions": {
            "ndcg_at_10": "Ranking quality at the first 10 recommendations; higher is better.",
            "map": "Mean average precision across ranking groups; higher is better.",
            "mrr": (
                "Expected reciprocal rank of the first relevant product; equal-score "
                "ties are averaged over possible orderings rather than resolved by row order."
            ),
            "tie_handling": (
                "Precision, Recall, F1, NDCG, MAP, and MRR average over equal-score "
                "orderings; metrics do not depend on CSV row order."
            ),
            "popularity_baseline": (
                "Ranks products by interaction counts in the training split only; "
                "products not seen during training receive a score of zero."
            ),
            "roc_auc": "Relevant-vs-not ranking discrimination; 0.5 is chance, 1.0 is perfect.",
            "latency_seconds_per_100_rows": "Time to score up to 100 test rows.",
            "peak_memory_mb": "Python-traced peak memory while scoring the efficiency sample.",
        },
    }
    output = Path(output_path)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        json.dump(_json_safe(report), destination, indent=2, allow_nan=False)
    logger.info("Per-dataset comparison written to %s", output)
    return report


if __name__ == "__main__":
    evaluate_available_datasets()
