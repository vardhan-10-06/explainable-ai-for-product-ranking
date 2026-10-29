"""
preprocessing/data_loader.py
=============================
Dataset-agnostic loader. Each supported public dataset has its own native
schema; this module's job is to map every one of them onto a single common
internal schema (users, products, interactions, reviews) so the rest of
the pipeline never has to know which dataset it is looking at.

Supported datasets: amazon, retailrocket, movielens, yelp.

Trade-off note: rather than writing four fully bespoke pipelines (accurate
to each dataset's real column names but 4x the maintenance surface), we
normalize at this single boundary. This is the standard approach used by
production recommender systems that ingest heterogeneous source data.
When pointed at the synthetic generator output the loader is a passthrough;
when pointed at a real dump, only `_map_<dataset>()` needs a column mapping.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from common.config import get_config, get_logger

logger = get_logger(__name__)

SUPPORTED_DATASETS = {"amazon", "flipkart", "retailrocket", "movielens", "yelp"}


@dataclass
class RawTables:
    """Container for the four common-schema tables used downstream."""
    users: pd.DataFrame
    products: pd.DataFrame
    interactions: pd.DataFrame
    reviews: pd.DataFrame


class DatasetLoader:
    """
    Loads a dataset (real or synthetic) and normalizes it to the common
    schema: users.csv, products.csv, interactions.csv, reviews.csv.
    """

    def __init__(self, dataset: Optional[str] = None, raw_dir: Optional[str] = None):
        cfg = get_config()["data"]
        self.dataset = (dataset or cfg["dataset"]).lower()
        self.raw_dir = Path(raw_dir or cfg["raw_dir"])
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{self.dataset}'. Choose from {SUPPORTED_DATASETS}"
            )

    def load(self) -> RawTables:
        """
        Load and normalize the dataset named in the config.

        Returns:
            RawTables with users/products/interactions/reviews DataFrames
            in the common schema.

        Raises:
            FileNotFoundError: if the expected CSVs are not present under
                `raw_dir`. For real datasets, first run the dataset's
                official export/download step and place files under
                `data/raw/<dataset>/` with native filenames; the mapping
                functions below document the expected native columns.
        """
        loader_fn = {
            "amazon": self._load_common_schema,       # our generator mimics Amazon-style events
            "flipkart": self._load_flipkart,
            "retailrocket": self._load_retailrocket,
            "movielens": self._load_movielens,
            "yelp": self._load_yelp,
        }[self.dataset]
        logger.info("Loading dataset '%s' from %s", self.dataset, self.raw_dir)
        tables = loader_fn()
        self._validate(tables)
        return tables

    # ------------------------------------------------------------------ #
    # Common-schema passthrough (used by the synthetic generator and by
    # any dataset already exported into our canonical CSV format).
    # ------------------------------------------------------------------ #
    def _load_common_schema(self) -> RawTables:
        users = pd.read_csv(self.raw_dir / "users.csv")
        products = pd.read_csv(self.raw_dir / "products.csv")
        interactions = pd.read_csv(self.raw_dir / "interactions.csv")
        reviews_path = self.raw_dir / "reviews.csv"
        reviews = pd.read_csv(reviews_path) if reviews_path.exists() else pd.DataFrame(
            columns=["review_id", "user_id", "product_id", "rating", "text", "days_ago"]
        )
        return RawTables(users, products, interactions, reviews)

    # ------------------------------------------------------------------ #
    # Flipkart product catalog (flipkart_combined.csv).
    # Product ratings are the relevance labels; categories are ranking
    # groups because this source contains no customer-level interactions.
    # ------------------------------------------------------------------ #
    def _load_flipkart(self) -> RawTables:
        source = self.raw_dir / "flipkart_combined.csv"
        if not source.exists():
            raise FileNotFoundError(
                f"Flipkart CSV not found at {source}. Place flipkart_combined.csv "
                "in the configured data.raw_dir."
            )

        frame = pd.read_csv(source)
        required = {"Category", "Title", "Rating"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Flipkart CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        frame = frame.copy()
        frame["Category"] = frame["Category"].astype("string").str.strip().str.lower()
        frame["Title"] = frame["Title"].astype("string").str.strip()
        frame["Rating"] = pd.to_numeric(frame["Rating"], errors="coerce")
        frame = frame.dropna(subset=["Category", "Title"])
        frame = frame[(frame["Category"] != "") & (frame["Title"] != "")]

        links = frame.get("Product Link", pd.Series("", index=frame.index)).fillna("").astype(str)
        product_ids = links.str.extract(r"[?&]pid=([^&]+)", expand=False)
        fallback_ids = links.where(
            links.str.strip().ne(""),
            frame["Category"] + "|" + frame["Title"],
        ).map(
            lambda value: "flipkart-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
        )
        frame["product_id"] = product_ids.fillna(fallback_ids)

        price_column = frame.get("Price", pd.Series(index=frame.index, dtype="string"))
        prices = pd.to_numeric(
            price_column.astype("string").str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce",
        )
        review_column = frame.get(
            "Ratings & Reviews", pd.Series(index=frame.index, dtype="string")
        )
        review_counts = review_column.astype("string").str.extract(
            r"(\d[\d,]*)", expand=False
        ).str.replace(",", "", regex=False)

        timestamps = pd.to_datetime(
            frame.get("Timestamp", pd.Series(index=frame.index, dtype="string")),
            errors="coerce",
        )
        latest_timestamp = timestamps.max()
        if pd.isna(latest_timestamp):
            days_ago = pd.Series(0, index=frame.index, dtype="int64")
        else:
            days_ago = (latest_timestamp - timestamps).dt.days.fillna(0).astype("int64")

        frame["price"] = prices
        frame["num_reviews"] = pd.to_numeric(review_counts, errors="coerce").fillna(0)
        frame["days_ago"] = days_ago
        if "Key Features" not in frame.columns:
            frame["Key Features"] = ""
        products = frame.groupby("product_id", as_index=False, sort=False).agg(
            title=("Title", "first"),
            key_features=("Key Features", lambda values: next(
                (value for value in values.dropna() if str(value).strip()),
                "",
            )),
            category=("Category", "first"),
            price=("price", "median"),
            num_reviews=("num_reviews", "max"),
            launch_days_ago=("days_ago", "min"),
        )
        products["brand"] = products["title"].str.split().str[0].fillna("Unknown")
        products["avg_rating"] = 0.0
        products = products[[
            "product_id", "title", "category", "brand", "price", "avg_rating",
            "num_reviews", "launch_days_ago",
        ]]

        rated = frame.dropna(subset=["Rating"]).groupby(
            ["Category", "product_id"], as_index=False, sort=False
        ).agg(
            rating=("Rating", "mean"),
            days_ago=("days_ago", "min"),
        )
        interactions = pd.DataFrame({
            "user_id": rated["Category"],
            "product_id": rated["product_id"],
            "event_type": "view",
            "days_ago": rated["days_ago"],
            "rating": rated["rating"],
            "relevance": (rated["rating"] >= 4.0).astype("int8"),
        })
        users = pd.DataFrame({"user_id": sorted(frame["Category"].unique())})
        users["signup_days_ago"] = int(frame["days_ago"].max()) if not frame.empty else 0
        users["avg_session_len"] = 0.0

        reviews = pd.DataFrame(
            columns=["review_id", "user_id", "product_id", "rating", "text", "days_ago"]
        )
        return RawTables(users, products, interactions, reviews)

    # ------------------------------------------------------------------ #
    # RetailRocket (events.csv, item_properties.csv, category_tree.csv)
    # Native columns: events.csv -> timestamp, visitorid, event, itemid,
    #                 transactionid
    # ------------------------------------------------------------------ #
    def _load_retailrocket(self) -> RawTables:
        events_path = self.raw_dir / "events.csv"
        if not events_path.exists():
            logger.warning(
                "RetailRocket raw files not found at %s; falling back to common-schema "
                "CSVs (use data/sample_data_generator.py to create them).",
                events_path,
            )
            return self._load_common_schema()

        events = pd.read_csv(events_path)
        event_map = {"view": "view", "addtocart": "cart", "transaction": "purchase"}
        interactions = pd.DataFrame({
            "user_id": events["visitorid"].astype(str),
            "product_id": events["itemid"].astype(str),
            "event_type": events["event"].map(event_map).fillna("view"),
            "days_ago": self._epoch_ms_to_days_ago(events["timestamp"]),
            "rating": None,
        })
        users = pd.DataFrame({"user_id": interactions["user_id"].unique()})
        users["signup_days_ago"] = interactions["days_ago"].max()
        users["avg_session_len"] = 5.0

        products = pd.DataFrame({"product_id": interactions["product_id"].unique()})
        products["category"] = "Unknown"
        products["brand"] = "Unknown"
        products["price"] = 0.0
        products["avg_rating"] = 0.0
        products["num_reviews"] = 0
        products["launch_days_ago"] = interactions["days_ago"].max()

        reviews = pd.DataFrame(columns=["review_id", "user_id", "product_id", "rating", "text", "days_ago"])
        return RawTables(users, products, interactions, reviews)

    # ------------------------------------------------------------------ #
    # MovieLens (ratings.csv, movies.csv)
    # Native columns: ratings.csv -> userId, movieId, rating, timestamp
    #                 movies.csv  -> movieId, title, genres
    # ------------------------------------------------------------------ #
    def _load_movielens(self) -> RawTables:
        ratings_path = self.raw_dir / "ratings.csv"
        movies_path = self.raw_dir / "movies.csv"
        if not (ratings_path.exists() and movies_path.exists()):
            logger.warning(
                "MovieLens raw files not found at %s; falling back to common-schema CSVs.",
                self.raw_dir,
            )
            return self._load_common_schema()

        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)

        interactions = pd.DataFrame({
            "user_id": ratings["userId"].astype(str),
            "product_id": ratings["movieId"].astype(str),
            "event_type": "purchase",  # a rating implies consumption
            "days_ago": self._epoch_s_to_days_ago(ratings["timestamp"]),
            "rating": ratings["rating"],
        })
        users = pd.DataFrame({"user_id": interactions["user_id"].unique()})
        users["signup_days_ago"] = interactions["days_ago"].max()
        users["avg_session_len"] = 5.0

        products = movies.rename(columns={"movieId": "product_id", "genres": "category"})
        products["product_id"] = products["product_id"].astype(str)
        products["category"] = products["category"].astype(str).str.split("|").str[0]
        products["brand"] = "Studio_Unknown"
        products["price"] = 0.0
        agg = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        agg.columns = ["product_id", "avg_rating", "num_reviews"]
        agg["product_id"] = agg["product_id"].astype(str)
        products = products.merge(agg, on="product_id", how="left")
        products["avg_rating"] = products["avg_rating"].fillna(0.0)
        products["num_reviews"] = products["num_reviews"].fillna(0).astype(int)
        products["launch_days_ago"] = interactions["days_ago"].max()
        products = products[[
            "product_id", "category", "brand", "price", "avg_rating",
            "num_reviews", "launch_days_ago",
        ]]

        reviews = pd.DataFrame(columns=["review_id", "user_id", "product_id", "rating", "text", "days_ago"])
        return RawTables(users, products, interactions, reviews)

    # ------------------------------------------------------------------ #
    # Yelp (yelp_academic_dataset_review.json, ..._business.json)
    # ------------------------------------------------------------------ #
    def _load_yelp(self) -> RawTables:
        review_path = self.raw_dir / "yelp_academic_dataset_review.json"
        business_path = self.raw_dir / "yelp_academic_dataset_business.json"
        if not (review_path.exists() and business_path.exists()):
            logger.warning(
                "Yelp raw files not found at %s; falling back to common-schema CSVs.",
                self.raw_dir,
            )
            return self._load_common_schema()

        reviews_raw = pd.read_json(review_path, lines=True)
        business_raw = pd.read_json(business_path, lines=True)

        interactions = pd.DataFrame({
            "user_id": reviews_raw["user_id"],
            "product_id": reviews_raw["business_id"],
            "event_type": "purchase",
            "days_ago": self._date_to_days_ago(reviews_raw["date"]),
            "rating": reviews_raw["stars"],
        })
        users = pd.DataFrame({"user_id": interactions["user_id"].unique()})
        users["signup_days_ago"] = interactions["days_ago"].max()
        users["avg_session_len"] = 5.0

        products = business_raw.rename(columns={
            "business_id": "product_id", "categories": "category", "stars": "avg_rating",
            "review_count": "num_reviews",
        })
        products["category"] = products["category"].astype(str).str.split(",").str[0]
        products["brand"] = products.get("name", "Unknown")
        products["price"] = 0.0
        products["launch_days_ago"] = interactions["days_ago"].max()
        products = products[[
            "product_id", "category", "brand", "price", "avg_rating",
            "num_reviews", "launch_days_ago",
        ]]

        reviews = pd.DataFrame({
            "review_id": reviews_raw["review_id"],
            "user_id": reviews_raw["user_id"],
            "product_id": reviews_raw["business_id"],
            "rating": reviews_raw["stars"],
            "text": reviews_raw["text"],
            "days_ago": interactions["days_ago"],
        })
        return RawTables(users, products, interactions, reviews)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _epoch_ms_to_days_ago(series: pd.Series) -> pd.Series:
        now_ms = series.max()
        return ((now_ms - series) / (1000 * 60 * 60 * 24)).round().astype(int)

    @staticmethod
    def _epoch_s_to_days_ago(series: pd.Series) -> pd.Series:
        now_s = series.max()
        return ((now_s - series) / (60 * 60 * 24)).round().astype(int)

    @staticmethod
    def _date_to_days_ago(series: pd.Series) -> pd.Series:
        dates = pd.to_datetime(series)
        return (dates.max() - dates).dt.days

    @staticmethod
    def _validate(tables: RawTables) -> None:
        """Sanity-check that required columns exist before returning."""
        required = {
            "users": {"user_id"},
            "products": {"product_id", "category"},
            "interactions": {"user_id", "product_id", "event_type", "days_ago"},
        }
        for name, cols in required.items():
            df = getattr(tables, name)
            missing = cols - set(df.columns)
            if missing:
                raise ValueError(f"Loaded '{name}' table is missing required columns: {missing}")
