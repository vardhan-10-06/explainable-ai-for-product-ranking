"""
feature_engineering/features.py
=================================
Builds the full feature matrix used by the ranking models from the
processed users/products/interactions/reviews tables plus (optionally)
knowledge-graph-derived embeddings.

Feature families implemented (per the project spec):
  * User features        - activity level, tenure, session behaviour
  * Product features      - price, rating, category/brand encodings
  * Interaction features   - user x product event counts/recency
  * Temporal features      - exponential recency decay
  * Behavioral features    - event-type propensities per user
  * Popularity features    - product popularity percentile
  * Review statistics      - review sentiment proxy, review count/rating
  * Embedding features      - SentenceTransformer text embeddings (reviews)
  * Graph features          - degree / PageRank / KG embedding (optional)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from common.config import get_config, get_logger

logger = get_logger(__name__)

TARGET_PROXY_FEATURES = {"pair_has_purchase"}

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy dependency
    _SBERT_AVAILABLE = False


class FeatureEngineer:
    """
    Computes the full feature matrix for (user, product) pairs used to
    train and score the ranking models.
    """

    def __init__(self, graph_features: Optional[pd.DataFrame] = None):
        """
        Args:
            graph_features: Optional DataFrame indexed by product_id with
                precomputed knowledge-graph features (degree, pagerank,
                embedding_0..N). If None, graph features default to zeros
                so the pipeline still runs without a KG backend configured.
        """
        self.cfg = get_config()
        self.graph_features = graph_features
        self._sbert_model = None
        self._user_features: Optional[pd.DataFrame] = None
        self._product_features: Optional[pd.DataFrame] = None
        self._review_features: Optional[pd.DataFrame] = None
        self._interaction_features: Optional[pd.DataFrame] = None
        if self.cfg["features"]["use_review_text_features"] and _SBERT_AVAILABLE:
            try:
                self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:  # pragma: no cover - network/model download issues
                logger.warning("Could not load SentenceTransformer (%s); using bag-of-words fallback.", exc)

    # ------------------------------------------------------------------ #
    def build_user_features(self, users: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-user activity/behavioral statistics."""
        counts = interactions.groupby("user_id")["event_type"].count().rename("total_events")
        event_pivot = (
            interactions.pivot_table(index="user_id", columns="event_type", values="product_id", aggfunc="count")
            .fillna(0)
        )
        event_pivot.columns = [f"event_share_{c}" for c in event_pivot.columns]
        event_pivot = event_pivot.div(event_pivot.sum(axis=1).replace(0, 1), axis=0)

        feats = users.set_index("user_id").join(counts).join(event_pivot).fillna(0)
        keep_cols = [
            c for c in feats.columns
            if c.startswith("event_share_") or c in {
                "signup_days_ago_scaled", "avg_session_len_scaled", "total_events",
            }
        ]
        return feats[keep_cols].reset_index()

    def build_product_features(self, products: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
        """Attach catalog, text, popularity, and graph features to products."""
        pop = interactions.groupby("product_id")["event_type"].count().rename("interaction_count")
        products = products.set_index("product_id").join(pop).fillna(0)
        products["popularity_percentile"] = products["interaction_count"].rank(pct=True)

        if "title" in products.columns or "key_features" in products.columns:
            titles = products.get("title", pd.Series("", index=products.index)).fillna("").astype(str)
            key_features = products.get(
                "key_features", pd.Series("", index=products.index)
            ).fillna("").astype(str)
            descriptions = titles.str.cat(key_features, sep=" ")
            vectorizer = HashingVectorizer(
                n_features=128,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                stop_words="english",
            )
            text_features = vectorizer.transform(descriptions).toarray()
            text_columns = [f"product_text_{index:03d}" for index in range(text_features.shape[1])]
            products = pd.concat(
                [
                    products,
                    pd.DataFrame(
                        text_features, index=products.index, columns=text_columns
                    ),
                ],
                axis=1,
            )

        if self.graph_features is not None:
            products = products.join(self.graph_features, how="left")
            graph_cols = [c for c in self.graph_features.columns]
            products[graph_cols] = products[graph_cols].fillna(0.0)
        else:
            products["graph_degree"] = 0.0
            products["graph_pagerank"] = 0.0

        keep_cols = [
            c for c in products.columns
            if c in {
                "category_encoded", "brand_encoded", "price_scaled", "avg_rating_scaled",
                "num_reviews_scaled", "launch_days_ago_scaled", "popularity_percentile",
                "interaction_count",
            } or c.startswith("graph_") or c.startswith("kg_emb_")
            or c.startswith("product_text_")
        ]
        return products[keep_cols].reset_index()

    def build_review_features(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-product review statistics and (optionally) a mean
        text-embedding vector summarizing review sentiment/content.
        """
        if reviews.empty:
            return pd.DataFrame({
                "product_id": pd.Series(dtype="object"),
                "review_avg_rating": pd.Series(dtype="float64"),
                "review_count": pd.Series(dtype="int64"),
            })

        stats = reviews.groupby("product_id").agg(
            review_avg_rating=("rating", "mean"),
            review_count=("rating", "count"),
        ).reset_index()

        if self._sbert_model is not None:
            embeddings = self._embed_reviews_sbert(reviews)
        else:
            embeddings = self._embed_reviews_bow(reviews)

        return stats.merge(embeddings, on="product_id", how="left")

    def _embed_reviews_sbert(self, reviews: pd.DataFrame) -> pd.DataFrame:
        grouped_text = reviews.groupby("product_id")["text"].apply(lambda s: " ".join(s.astype(str))[:2000])
        vectors = self._sbert_model.encode(grouped_text.tolist(), show_progress_bar=False)
        dim = self.cfg["features"]["embedding_dim"]
        vectors = vectors[:, :dim] if vectors.shape[1] >= dim else np.pad(
            vectors, ((0, 0), (0, dim - vectors.shape[1]))
        )
        cols = [f"review_emb_{i}" for i in range(dim)]
        emb_df = pd.DataFrame(vectors, columns=cols)
        emb_df.insert(0, "product_id", grouped_text.index)
        return emb_df

    def _embed_reviews_bow(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """Deterministic offline fallback when SentenceTransformer isn't available."""
        dim = self.cfg["features"]["embedding_dim"]
        grouped_text = reviews.groupby("product_id")["text"].apply(lambda s: " ".join(s.astype(str)))

        def _hash_embed(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.normal(0, 1, size=dim)

        vectors = np.stack([_hash_embed(t) for t in grouped_text])
        cols = [f"review_emb_{i}" for i in range(dim)]
        emb_df = pd.DataFrame(vectors, columns=cols)
        emb_df.insert(0, "product_id", grouped_text.index)
        return emb_df

    def build_interaction_features(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute (user, product) pair-level features: interaction count,
        recency-weighted affinity, and whether the pair has a purchase.
        """
        halflife = self.cfg["features"]["temporal_decay_halflife_days"]
        interactions = interactions.copy()
        interactions["recency_weight"] = 0.5 ** (interactions["days_ago"] / halflife)

        pair_feats = interactions.groupby(["user_id", "product_id"]).agg(
            pair_event_count=("event_type", "count"),
            pair_recency_score=("recency_weight", "sum"),
            pair_has_purchase=("event_type", lambda s: int((s == "purchase").any())),
        ).reset_index()
        if "relevance" in interactions.columns:
            labels = interactions.groupby(["user_id", "product_id"])["relevance"].max()
            pair_feats = pair_feats.merge(
                labels.rename("label"), on=["user_id", "product_id"], how="left"
            )
        return pair_feats

    # ------------------------------------------------------------------ #
    def build_full_feature_matrix(
        self,
        users: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
        reviews: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join every feature family into one (user, product) level matrix
        suitable for training the ranking models.

        Returns:
            DataFrame with one row per observed (user_id, product_id) pair
            plus every engineered feature column, and a `label` column
            (1 = purchase occurred, 0 otherwise) for supervised ranking.
        """
        user_feats = self.build_user_features(users, interactions)
        product_feats = self.build_product_features(products, interactions)
        review_feats = self.build_review_features(reviews)
        pair_feats = self.build_interaction_features(interactions)
        self._user_features = user_feats
        self._product_features = product_feats
        self._review_features = review_feats
        self._interaction_features = pair_feats

        matrix = pair_feats.merge(user_feats, on="user_id", how="left")
        matrix = matrix.merge(product_feats, on="product_id", how="left")
        matrix = matrix.merge(review_feats, on="product_id", how="left")
        matrix = matrix.fillna(0.0)
        if "label" not in matrix.columns:
            matrix["label"] = matrix["pair_has_purchase"]

        logger.info("Built feature matrix with shape %s", matrix.shape)
        return matrix

    def build_candidate_feature_matrix(
        self, user_id: str, product_ids: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Build feature rows for a user's catalog candidates, including unseen products."""
        if any(
            frame is None
            for frame in (
                self._user_features,
                self._product_features,
                self._review_features,
                self._interaction_features,
            )
        ):
            raise RuntimeError("Build the full feature matrix before generating ranking candidates.")

        user_rows = self._user_features[self._user_features["user_id"] == user_id]
        if user_rows.empty:
            return pd.DataFrame()

        products = self._product_features
        if product_ids is not None:
            products = products[products["product_id"].isin(product_ids)]
        if products.empty:
            return pd.DataFrame()

        candidates = products[["product_id"]].assign(user_id=user_id)
        matrix = candidates.merge(self._interaction_features, on=["user_id", "product_id"], how="left")
        matrix = matrix.merge(user_rows, on="user_id", how="left")
        matrix = matrix.merge(products, on="product_id", how="left")
        matrix = matrix.merge(self._review_features, on="product_id", how="left")
        return matrix.fillna(0.0)
