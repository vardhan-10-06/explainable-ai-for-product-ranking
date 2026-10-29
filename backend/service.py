"""
backend/service.py
====================
Business-logic layer, separated from the FastAPI route handlers per the
project's "separate business logic from APIs" requirement. Lazily loads
and caches the trained ensemble, processed data, knowledge graph, and
explainability components so routers stay thin.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from common.config import get_config, get_logger
from common.config import PROJECT_ROOT
from evaluation.ecs import ExplainabilityConfidenceScore
from explainability.rank_explainer import RankExplainer
from explainability.shap_explainer import ShapRankingExplainer
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from llm.explanation_generator import ExplanationGenerator
from preprocessing.data_loader import DatasetLoader
from preprocessing.preprocess import Preprocessor
from ranking.ensemble import HybridRankingEnsemble
from feature_engineering.features import FeatureEngineer, TARGET_PROXY_FEATURES

logger = get_logger(__name__)

ID_COLS = ["user_id", "product_id"]
LABEL_COL = "label"

FRIENDLY_FEATURE_NAMES = {
    "price_scaled": "listed price",
    "num_reviews_scaled": "number of customer ratings and reviews",
    "avg_rating_scaled": "customer rating",
    "review_avg_rating": "customer review feedback",
    "review_count": "number of written reviews",
    "category_encoded": "category-level pattern",
    "brand_encoded": "brand-level pattern",
    "popularity_percentile": "product popularity in this catalog",
    "interaction_count": "shopper activity recorded for this product",
    "total_events": "category activity in this dataset",
    "launch_days_ago_scaled": "how recently the product was listed",
    "graph_degree": "connections to other catalog items",
    "graph_pagerank": "importance within the product catalogue",
}


def _friendly_feature_name(feature: str) -> str:
    """Translate internal feature identifiers into wording for shoppers."""
    if feature in FRIENDLY_FEATURE_NAMES:
        return FRIENDLY_FEATURE_NAMES[feature]
    if feature.startswith("kg_emb_") or feature.startswith("review_emb_"):
        return "product and review patterns"
    if feature.startswith("product_text_"):
        return "product name and specification terms"
    if feature.startswith("event_share_"):
        return "category-level shopping patterns"
    return feature.replace("_scaled", "").replace("_", " ")


def _plain_language_summary(
    features: List[Dict], selected_title: str, compared_title: Optional[str] = None
) -> str:
    """Summarize only the measured feature contributions; do not claim product quality."""
    if compared_title is not None:
        strongest = features[:3]
        if not strongest:
            return (
                f"The model compared {selected_title} with {compared_title}, but no single "
                "catalog signal clearly explained the score difference."
            )
        differences = [
            f"{_friendly_feature_name(item['feature'])} favored "
            f"{selected_title if item['favors'] == 'product_a' else compared_title}"
            for item in strongest
        ]
        return (
            f"The main differences between these products were: {'; '.join(differences)}. "
            "These signals explain the model's ordering; they do not guarantee which "
            "product is the better choice for you."
        )

    supporting = [
        _friendly_feature_name(item["feature"])
        for item in features
        if item.get("shap_value", 0) > 0
    ][:3]
    lowering = [
        _friendly_feature_name(item["feature"])
        for item in features
        if item.get("shap_value", 0) < 0
    ][:2]
    if not supporting:
        return (
            "The product's position comes from the model's combined catalog signals; "
            "no single factor clearly stood out. Use the listed price and product details "
            "to decide whether it suits your needs."
        )
    summary = (
        f"For {selected_title}, the main signals supporting its position were "
        f"{', '.join(supporting)}."
    )
    if lowering:
        summary += f" Signals that pulled the score down included {', '.join(lowering)}."
    return summary + " These are ranking signals, not a guarantee of product quality or fit."


class RankingService:
    """
    Loads all heavyweight artifacts once and exposes simple methods for
    the API routers: search, rank, explain, and evaluate.

    Raises a clear RuntimeError (surfaced to the API caller as a 503) if
    called before `ranking/train.py` has produced a trained ensemble --
    this service never silently trains a model on a live API request.
    """

    def __init__(self, ensemble_path: str = str(PROJECT_ROOT / "results" / "hybrid_ensemble.pkl")):
        self.cfg = get_config()
        self.ensemble_path = ensemble_path
        self.ensemble: Optional[HybridRankingEnsemble] = None
        self.feature_cols: List[str] = []
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.products: Optional[pd.DataFrame] = None
        self.kg_builder: Optional[KnowledgeGraphBuilder] = None
        self.shap_explainer: Optional[ShapRankingExplainer] = None
        self.rank_explainer: Optional[RankExplainer] = None
        self.explanation_generator = ExplanationGenerator()
        self.ecs = ExplainabilityConfidenceScore()
        self._loaded = False
        self._explainability_loaded = False

    def ensure_loaded(self) -> None:
        """Load all artifacts on first use; no-op on subsequent calls."""
        if self._loaded:
            return

        path = Path(self.ensemble_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise RuntimeError(
                f"No trained ensemble found at {path}. Run `python -m ranking.train` first."
            )
        with open(path, "rb") as fh:
            artifact = pickle.load(fh)
        self.ensemble, self.feature_cols = artifact["ensemble"], artifact["feature_cols"]
        leaked_features = TARGET_PROXY_FEATURES.intersection(self.feature_cols)
        if leaked_features:
            raise RuntimeError(
                "The saved ranking model contains target-leaking features "
                f"({', '.join(sorted(leaked_features))}). Retrain it with "
                "`python -m ranking.train` before starting the API."
            )

        raw_dir = PROJECT_ROOT / self.cfg["data"]["raw_dir"]
        raw = DatasetLoader(raw_dir=str(raw_dir)).load()
        processed = Preprocessor().run(raw)
        self.products = processed.products

        self.kg_builder = KnowledgeGraphBuilder()
        self.kg_builder.build(processed.users, processed.products, processed.interactions, processed.reviews)
        graph_feats = self.kg_builder.compute_graph_features()
        kg_embeddings = self.kg_builder.compute_graph_embeddings()
        graph_feats = graph_feats.join(kg_embeddings, how="left").fillna(0.0)

        fe = FeatureEngineer(graph_features=graph_feats)
        self.feature_matrix = fe.build_full_feature_matrix(
            processed.users, processed.products, processed.interactions, processed.reviews
        )
        self.feature_engineer = fe
        if self.ensemble.strategy == "weighted" and not getattr(self.ensemble, "score_statistics_", None):
            self.ensemble.calibrate(self.feature_matrix[self.feature_cols])

        self._loaded = True
        logger.info("RankingService fully loaded and ready.")

    def ensure_explainability_loaded(self) -> None:
        """Build SHAP explainers only when an explanation is requested.

        Ranking and catalog requests do not need SHAP or KernelExplainer.
        Deferring this expensive setup keeps the normal dashboard workflow
        responsive while preserving the full explanation path on demand.
        """
        self.ensure_loaded()
        if self._explainability_loaded:
            return
        background = self.feature_matrix[self.feature_cols]
        self.shap_explainer = ShapRankingExplainer(self.ensemble, background)
        self.rank_explainer = RankExplainer(self.shap_explainer, self.kg_builder)
        self._explainability_loaded = True
        logger.info("Explainability components loaded.")

    # ------------------------------------------------------------------ #
    def search_products(self, query: str, category: Optional[str], top_k: int) -> List[Dict]:
        """Simple substring/category search over the product catalog."""
        self.ensure_loaded()
        df = self.products
        mask = (
            df["product_id"].astype(str).str.contains(
                query, case=False, na=False, regex=False
            )
            | df.get("title", pd.Series("", index=df.index)).astype(str).str.contains(
                query, case=False, na=False, regex=False
            )
            | df["category"].astype(str).str.contains(
                query, case=False, na=False, regex=False
            )
            | df["brand"].astype(str).str.contains(
                query, case=False, na=False, regex=False
            )
        )
        if category:
            mask &= (
                df["category"].astype(str).str.strip().str.lower()
                == category.strip().lower()
            )
        results = df[mask].head(top_k)
        return results.to_dict(orient="records")

    def _resolve_user_candidates(
        self, user_id: str, candidate_product_ids: Optional[List[str]]
    ) -> tuple[str, Optional[List[str]]]:
        """Treat Flipkart category IDs as ranking groups and restrict their catalog."""
        if self.cfg["data"]["dataset"] != "flipkart":
            return user_id, candidate_product_ids
        if self.products is None:
            raise RuntimeError("Product data is unavailable after loading the ranking service.")

        category = user_id.strip().lower()
        category_products = self.products.loc[
            self.products["category"].astype(str).str.strip().str.lower() == category,
            "product_id",
        ].astype(str)
        if candidate_product_ids is not None:
            requested = set(candidate_product_ids)
            category_products = category_products[category_products.isin(requested)]
        return category, category_products.tolist()

    def rank_for_user(self, user_id: str, candidate_product_ids: Optional[List[str]], top_k: int) -> List[Dict]:
        """Score catalog candidates for a user and return the top-k ranked list."""
        self.ensure_loaded()
        if self.feature_engineer is None:
            raise RuntimeError("Ranking feature data is unavailable after loading the model.")
        user_id, candidate_product_ids = self._resolve_user_candidates(
            user_id, candidate_product_ids
        )
        user_rows = self.feature_engineer.build_candidate_feature_matrix(
            user_id, candidate_product_ids
        )
        if user_rows.empty:
            return []

        if candidate_product_ids is None:
            user_rows = user_rows[user_rows["pair_has_purchase"] == 0]
        if user_rows.empty:
            return []

        X = user_rows[self.feature_cols]
        scores = self.ensemble.predict(X)
        user_rows = user_rows.assign(rank_score=scores).sort_values("rank_score", ascending=False).head(top_k)

        product_details = self.products.set_index("product_id")
        ranked = []
        for row in user_rows.itertuples(index=False):
            product = product_details.loc[row.product_id]
            title = product.get("title", row.product_id)
            price = product.get("price")
            brand = product.get("brand")
            num_reviews = product.get("num_reviews")
            ranked.append({
                "product_id": row.product_id,
                "rank_score": float(row.rank_score),
                "rank_position": len(ranked) + 1,
                "title": str(title) if pd.notna(title) else row.product_id,
                "price": float(price) if pd.notna(price) else None,
                "brand": str(brand) if pd.notna(brand) else None,
                "num_reviews": int(num_reviews) if pd.notna(num_reviews) else None,
            })
        return ranked

    def _product_display_details(self, product_id: Optional[str]) -> Dict[str, Optional[object]]:
        """Get display-only catalog details without adding them to model features."""
        if not product_id or self.products is None:
            return {"title": None, "price": None}
        product = self.products[self.products["product_id"] == product_id]
        if product.empty:
            return {"title": None, "price": None}
        row = product.iloc[0]
        price = row.get("price")
        title = row.get("title", product_id)
        return {
            "title": str(title) if pd.notna(title) else product_id,
            "price": float(price) if pd.notna(price) else None,
        }

    def explain(self, user_id: str, product_id: str, compare_to_product_id: Optional[str] = None) -> Dict:
        """Produce a full verified explanation (+ ECS score) for one ranking decision."""
        self.ensure_explainability_loaded()
        if self.feature_engineer is None:
            raise RuntimeError("Ranking feature data is unavailable after loading the model.")
        user_id, candidate_product_ids = self._resolve_user_candidates(user_id, None)
        user_candidates = self.feature_engineer.build_candidate_feature_matrix(
            user_id, candidate_product_ids
        )
        row = user_candidates[user_candidates["product_id"] == product_id]
        if row.empty:
            raise ValueError(f"No feature row found for user_id={user_id}, product_id={product_id}")

        X_row = row[self.feature_cols]
        rank_score = float(self.ensemble.predict(X_row)[0])

        compare_to = None
        compare_score = None
        if compare_to_product_id:
            row_b = user_candidates[user_candidates["product_id"] == compare_to_product_id]
            if row_b.empty:
                raise ValueError(
                    f"No feature row found for user_id={user_id}, product_id={compare_to_product_id}"
                )
            compare_to = {"product_id": compare_to_product_id, "x_row": row_b[self.feature_cols]}
            compare_score = float(self.ensemble.predict(compare_to["x_row"])[0])

        package = self.rank_explainer.explain(user_id, product_id, X_row, rank_score, compare_to=compare_to)

        if compare_to is not None:
            generated = self.explanation_generator.generate_comparison_explanation(package, compare_to_product_id)
        else:
            generated = self.explanation_generator.generate_single_explanation(package)

        product_details = self._product_display_details(product_id)
        compared_details = self._product_display_details(compare_to_product_id)
        explanation_features = package.comparison or package.top_shap_features
        plain_language_summary = _plain_language_summary(
            explanation_features,
            str(product_details["title"] or product_id),
            str(compared_details["title"]) if compare_to_product_id else None,
        )

        all_shap = self.shap_explainer.explain_instances(X_row).iloc[0].values
        feature_indices = {feature: index for index, feature in enumerate(self.feature_cols)}
        top_k_idx = [
            feature_indices[feature["feature"]]
            for feature in package.top_shap_features
            if feature["feature"] in feature_indices
        ]
        ecs_result = self.ecs.compute(
            all_shap_values=all_shap,
            top_k_indices=top_k_idx,
            graph_path=package.graph_path,
            hallucination_confidence=generated.confidence_score,
            features_run_1=[f["feature"] for f in package.top_shap_features],
        )

        return {
            "product_id": product_id,
            "product_title": product_details["title"],
            "price": product_details["price"],
            "compared_product_id": compare_to_product_id,
            "compared_product_title": compared_details["title"],
            "compared_price": compared_details["price"],
            "rank_score": rank_score,
            "compared_rank_score": compare_score,
            "score_difference": rank_score - compare_score if compare_score is not None else None,
            "explanation_text": generated.text,
            "plain_language_summary": plain_language_summary,
            "confidence_score": generated.confidence_score,
            "ecs_score": ecs_result.score,
            "passed_verification": generated.verification.passed,
            "top_shap_features": package.top_shap_features,
            "comparison_features": package.comparison or [],
            "graph_path": package.graph_path,
        }


@lru_cache(maxsize=1)
def get_ranking_service() -> RankingService:
    """FastAPI dependency: return the process-wide singleton RankingService."""
    return RankingService()
