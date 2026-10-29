"""
llm/prompt_templates.py
=========================
Prompt templates for turning a structured ExplanationPackage into a
grounded natural-language explanation request for the LLM.

Design principle: the prompt explicitly instructs the model to ONLY use
the evidence provided (SHAP features + KG path) and to avoid speculation,
which is the primary lever for hallucination reduction at the prompting
level (the hallucination_detector module then verifies compliance).
"""
from __future__ import annotations

from typing import Dict, List

SYSTEM_PROMPT = """You are an explainability assistant for a product ranking \
system. You must explain ranking decisions ONLY using the evidence given to \
you: SHAP feature contributions and knowledge-graph relationship paths. \
Do not invent facts, prices, reviews, or relationships that are not present \
in the evidence. If the evidence is weak or sparse, say so plainly instead \
of guessing. Keep explanations concise (2-4 sentences) and in plain, \
user-friendly language -- avoid ML jargon like "SHAP" or "embedding"."""


def build_single_explanation_prompt(
    product_id: str,
    rank_score: float,
    top_shap_features: List[Dict],
    graph_path: List[Dict],
) -> str:
    """
    Build the user-turn prompt for explaining a single product's ranking.

    Args:
        product_id: The product being explained.
        rank_score: The model's predicted relevance score.
        top_shap_features: Output of ShapRankingExplainer.local_explanation().
        graph_path: Output of KnowledgeGraphBuilder.explain_path().

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    feature_lines = "\n".join(
        f"- {f['feature']} ({f['direction'].replace('_', ' ')}, strength={f['shap_value']:.3f})"
        for f in top_shap_features
    ) or "- No significant feature contributions found."

    if graph_path:
        path_line = " -> ".join(f"{edge['from']} --[{edge['relation']}]--> {edge['to']}" for edge in graph_path)
    else:
        path_line = "No direct knowledge-graph path found between this user and product."

    return f"""Product: {product_id}
Predicted relevance score: {rank_score:.4f}

Evidence - top contributing features (from the ranking model):
{feature_lines}

Evidence - knowledge graph relationship path:
{path_line}

Using ONLY the evidence above, explain in plain language why this product \
was ranked highly for this user. If the evidence is sparse, acknowledge \
that explicitly rather than fabricating a reason."""


def build_comparison_prompt(
    product_a: str, product_b: str, comparison_features: List[Dict]
) -> str:
    """
    Build the user-turn prompt for explaining why product_a outranked
    product_b.

    Args:
        product_a: The higher-ranked product id.
        product_b: The lower-ranked product id.
        comparison_features: Output of ShapRankingExplainer.explain_pairwise_ranking().

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    lines = "\n".join(
        f"- {f['feature']} favors {'Product A' if f['favors'] == 'product_a' else 'Product B'} "
        f"(difference strength={abs(f['delta_shap']):.3f})"
        for f in comparison_features
    ) or "- No meaningfully differentiating features found."

    return f"""Product A: {product_a}
Product B: {product_b}

Evidence - features differentiating the two rankings:
{lines}

Using ONLY the evidence above, explain in plain language which product is \
favored by the measured feature contributions and why. Do not assume Product A \
ranked above Product B: the evidence may favor either product. Be explicit \
about which product the strongest differences favor."""
