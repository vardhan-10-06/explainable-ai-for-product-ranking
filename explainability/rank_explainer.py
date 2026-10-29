"""
explainability/rank_explainer.py
===================================
Orchestrates a full ranking explanation for a (user, product) pair by
combining:
  * SHAP local feature attributions (explainability/shap_explainer.py)
  * Knowledge-graph evidence paths (knowledge_graph/graph_builder.py)

The output of `RankExplainer.explain()` is the structured "explanation
package" consumed by the LLM layer (llm/explanation_generator.py) to
produce the final natural-language explanation, and by the hallucination
detector to fact-check any claim the LLM makes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from common.config import get_logger
from explainability.shap_explainer import ShapRankingExplainer
from knowledge_graph.graph_builder import KnowledgeGraphBuilder

logger = get_logger(__name__)


@dataclass
class ExplanationPackage:
    """Structured evidence bundle for one ranking decision, ready for LLM consumption."""
    user_id: str
    product_id: str
    rank_score: float
    top_shap_features: List[Dict] = field(default_factory=list)
    graph_path: List[Dict] = field(default_factory=list)
    comparison: Optional[List[Dict]] = None   # populated when comparing against another product


class RankExplainer:
    """High-level explanation orchestrator combining SHAP and knowledge-graph evidence."""

    def __init__(self, shap_explainer: ShapRankingExplainer, kg_builder: KnowledgeGraphBuilder):
        self.shap_explainer = shap_explainer
        self.kg_builder = kg_builder

    def explain(
        self,
        user_id: str,
        product_id: str,
        x_row: pd.DataFrame,
        rank_score: float,
        compare_to: Optional[Dict] = None,
    ) -> ExplanationPackage:
        """
        Build a complete explanation package for why `product_id` received
        `rank_score` for `user_id`.

        Args:
            user_id: The user being served a ranking.
            product_id: The product whose ranking is being explained.
            x_row: The single-row feature vector for (user_id, product_id).
            rank_score: The ensemble's predicted relevance score.
            compare_to: Optional dict with keys {"product_id", "x_row"} for
                a second product, to additionally generate a pairwise
                RankSHAP-style comparison ("why A over B").

        Returns:
            An ExplanationPackage combining SHAP attributions and KG paths.
        """
        top_features = self.shap_explainer.local_explanation(x_row)
        graph_path = self.kg_builder.explain_path(user_id, product_id)

        comparison = None
        if compare_to is not None:
            comparison = self.shap_explainer.explain_pairwise_ranking(x_row, compare_to["x_row"])

        package = ExplanationPackage(
            user_id=user_id,
            product_id=product_id,
            rank_score=float(rank_score),
            top_shap_features=top_features,
            graph_path=graph_path,
            comparison=comparison,
        )
        logger.debug("Built explanation package for user=%s product=%s", user_id, product_id)
        return package
