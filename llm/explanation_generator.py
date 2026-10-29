"""
llm/explanation_generator.py
==============================
Top-level API for generating a verified, natural-language explanation of a
ranking decision. Wires together:
    ExplanationPackage (SHAP + KG evidence)
        -> prompt_templates
        -> LLMClient.generate()
        -> HallucinationDetector.verify()
    -> final explanation + confidence score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from common.config import get_logger
from explainability.rank_explainer import ExplanationPackage
from llm.hallucination_detector import HallucinationDetector, VerificationResult
from llm.llm_client import LLMClient
from llm.prompt_templates import build_comparison_prompt, build_single_explanation_prompt

logger = get_logger(__name__)


@dataclass
class GeneratedExplanation:
    """Final explanation output surfaced to the user/dashboard."""
    text: str
    confidence_score: float
    verification: VerificationResult
    raw_prompt: str


class ExplanationGenerator:
    """Generates and verifies natural-language ranking explanations."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.hallucination_detector = HallucinationDetector()

    def generate_single_explanation(self, package: ExplanationPackage) -> GeneratedExplanation:
        """
        Generate and verify an explanation for a single product's ranking.

        Args:
            package: ExplanationPackage produced by RankExplainer.explain().

        Returns:
            GeneratedExplanation with the final text, a confidence score,
            and the full verification result (for audit/debugging).
        """
        prompt = build_single_explanation_prompt(
            package.product_id, package.rank_score, package.top_shap_features, package.graph_path,
        )
        raw_text = self.llm_client.generate(prompt)
        verification = self.hallucination_detector.verify(
            explanation_text=raw_text,
            evidence_features=package.top_shap_features,
            evidence_graph_path=package.graph_path,
        )
        final_text = raw_text if verification.passed else self._safe_fallback(package)
        logger.info(
            "Generated explanation for product=%s (confidence=%.2f, passed=%s)",
            package.product_id, verification.confidence_score, verification.passed,
        )
        return GeneratedExplanation(
            text=final_text,
            confidence_score=verification.confidence_score,
            verification=verification,
            raw_prompt=prompt,
        )

    def generate_comparison_explanation(self, package: ExplanationPackage, product_b: str) -> GeneratedExplanation:
        """
        Generate and verify a pairwise "why A over B" explanation.

        Args:
            package: ExplanationPackage for product A, with `comparison`
                already populated (see RankExplainer.explain(compare_to=...)).
            product_b: The id of the lower-ranked product being compared against.

        Returns:
            GeneratedExplanation for the comparison.
        """
        if package.comparison is None:
            raise ValueError("ExplanationPackage.comparison is empty; call RankExplainer.explain(compare_to=...) first.")

        prompt = build_comparison_prompt(package.product_id, product_b, package.comparison)
        raw_text = self.llm_client.generate(prompt)
        verification = self.hallucination_detector.verify(
            explanation_text=raw_text,
            evidence_features=[
                {"feature": f["feature"], "shap_value": f["delta_shap"], "direction": f["favors"]}
                for f in package.comparison
            ],
            evidence_graph_path=package.graph_path,
        )
        final_text = (
            raw_text
            if verification.passed
            else self._safe_comparison_fallback(package, product_b)
        )
        return GeneratedExplanation(
            text=final_text,
            confidence_score=verification.confidence_score,
            verification=verification,
            raw_prompt=prompt,
        )

    @staticmethod
    def _safe_fallback(package: ExplanationPackage) -> str:
        """
        Deterministic, evidence-only fallback text used whenever the LLM's
        free-text output fails hallucination verification -- guarantees the
        user is never shown an unverified claim.
        """
        if not package.top_shap_features:
            return "This ranking is based on the model's overall score; no strongly dominant factor was found."
        top = package.top_shap_features[0]
        return (
            f"This ranking was primarily influenced by '{top['feature'].replace('_', ' ')}', "
            f"which {top['direction'].replace('_', ' ')} for this user."
        )

    @staticmethod
    def _safe_comparison_fallback(package: ExplanationPackage, product_b: str) -> str:
        """Describe the strongest pairwise evidence without asserting an unsupported winner."""
        if not package.comparison:
            return "There is not enough pairwise feature evidence to explain this score difference."
        top = package.comparison[0]
        favored = package.product_id if top["favors"] == "product_a" else product_b
        other = product_b if favored == package.product_id else package.product_id
        return (
            f"Compared with {other}, the available feature evidence most strongly favors "
            f"{favored} on '{top['feature'].replace('_', ' ')}'."
        )
