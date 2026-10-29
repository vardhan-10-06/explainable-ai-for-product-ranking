"""
evaluation/ecs.py
====================
Explainability Confidence Score (ECS) -- the novel research contribution
of this project.

--------------------------------------------------------------------------
Mathematical formulation
--------------------------------------------------------------------------
For a single ranking decision i, define five component scores, each
normalized to [0, 1]:

    F_i  - Feature Importance Confidence:
           F_i = sum_{f in topK(i)} |shap_f| / sum_{f in all features} |shap_f|
           (the share of total explanatory "mass" concentrated in the
           top-K features actually shown to the user; higher = a small,
           legible set of features explains most of the decision)

    G_i  - Knowledge Graph Evidence Score:
           G_i = min(1, hops_found^-1 * path_exists) combined with path
           length, i.e.:
                G_i = 1 / (1 + path_length_i)   if a KG path exists
                G_i = 0                          otherwise
           (shorter, existing graph paths are stronger structural evidence)

    L_i  - LLM Confidence:
           L_i in [0, 1], the LLM/template engine's own reported confidence
           in the generated explanation (e.g. derived from log-probs for a
           real LLM provider, or a fixed high value for the deterministic
           template engine since it cannot fabricate claims).

    V_i  - Fact Verification Score:
           V_i = HallucinationDetector.verify(...).confidence_score
           (0-1 score from evidence/claim overlap; see
           llm/hallucination_detector.py)

    C_i  - Explanation Consistency Score:
           C_i = Jaccard(topK_features_run_1, topK_features_run_2)
           measuring whether re-generating the explanation for the same
           instance (e.g. after a small perturbation or model refresh)
           yields the same key features -- i.e. explanation stability.

The final Explainability Confidence Score is the weighted sum:

    ECS_i = w_F * F_i + w_G * G_i + w_L * L_i + w_V * V_i + w_C * C_i

with weights w_F + w_G + w_L + w_V + w_C = 1 (configurable, see
configs/config.yaml -> ecs.weights; defaults: 0.25/0.25/0.2/0.2/0.1).

ECS in [0, 1]: values near 1 indicate a ranking decision whose explanation
is well-supported by feature importance concentration, structural
knowledge-graph evidence, LLM confidence, verified facts, and stability
under perturbation. Values near 0 indicate an explanation that should be
treated with skepticism (e.g. surfaced to a human reviewer, or replaced
with the deterministic fallback text in explanation_generator.py).
--------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from common.config import get_config, get_logger

logger = get_logger(__name__)


@dataclass
class ECSComponents:
    """The five normalized sub-scores that make up one ECS value."""
    feature_importance_score: float
    kg_evidence_score: float
    llm_confidence_score: float
    fact_verification_score: float
    explanation_consistency_score: float


@dataclass
class ECSResult:
    """Final ECS value plus its components, for transparency/auditing."""
    score: float
    components: ECSComponents


class ExplainabilityConfidenceScore:
    """Computes the Explainability Confidence Score (ECS) for a ranking decision."""

    def __init__(self):
        self.weights = get_config()["ecs"]["weights"]

    def compute_feature_importance_score(self, all_shap_values: np.ndarray, top_k_indices: List[int]) -> float:
        """
        F_i: share of total |SHAP| mass concentrated in the displayed top-K
        features. See module docstring for the full formula.
        """
        total_mass = np.abs(all_shap_values).sum()
        if total_mass == 0:
            return 0.0
        top_mass = np.abs(all_shap_values[top_k_indices]).sum()
        return float(min(1.0, top_mass / total_mass))

    def compute_kg_evidence_score(self, graph_path: List[Dict]) -> float:
        """G_i: inverse path-length score, 0 if no path exists. See module docstring."""
        if not graph_path:
            return 0.0
        path_length = len(graph_path)
        return float(1.0 / (1.0 + path_length))

    def compute_llm_confidence_score(self, llm_reported_confidence: Optional[float]) -> float:
        """
        L_i: the LLM/template engine's own confidence. Defaults to 0.8 when
        the provider doesn't expose a native confidence signal (e.g. the
        deterministic template engine, which by construction cannot
        fabricate claims, so a moderately high fixed prior is reasonable).
        """
        if llm_reported_confidence is None:
            return 0.8
        return float(max(0.0, min(1.0, llm_reported_confidence)))

    def compute_fact_verification_score(self, hallucination_confidence: float) -> float:
        """V_i: directly reuses HallucinationDetector's confidence_score output."""
        return float(max(0.0, min(1.0, hallucination_confidence)))

    def compute_consistency_score(self, features_run_1: List[str], features_run_2: List[str]) -> float:
        """C_i: Jaccard overlap between top-feature sets across two explanation runs."""
        set_a, set_b = set(features_run_1), set(features_run_2)
        union = set_a | set_b
        if not union:
            return 1.0
        return float(len(set_a & set_b) / len(union))

    def compute(
        self,
        all_shap_values: np.ndarray,
        top_k_indices: List[int],
        graph_path: List[Dict],
        hallucination_confidence: float,
        features_run_1: List[str],
        features_run_2: Optional[List[str]] = None,
        llm_reported_confidence: Optional[float] = None,
    ) -> ECSResult:
        """
        Compute the full ECS for one ranking decision.

        Args:
            all_shap_values: Full SHAP value vector for the instance.
            top_k_indices: Indices of the top-K features shown to the user.
            graph_path: KG evidence path (possibly empty).
            hallucination_confidence: Output of HallucinationDetector.verify().confidence_score.
            features_run_1: Top-feature names from the primary explanation run.
            features_run_2: Top-feature names from a repeated/perturbed run
                (for the consistency component); if None, consistency
                defaults to 1.0 (no repeated run available to compare against).
            llm_reported_confidence: Optional native LLM confidence signal.

        Returns:
            ECSResult with the final weighted score and all components.
        """
        components = ECSComponents(
            feature_importance_score=self.compute_feature_importance_score(all_shap_values, top_k_indices),
            kg_evidence_score=self.compute_kg_evidence_score(graph_path),
            llm_confidence_score=self.compute_llm_confidence_score(llm_reported_confidence),
            fact_verification_score=self.compute_fact_verification_score(hallucination_confidence),
            explanation_consistency_score=(
                self.compute_consistency_score(features_run_1, features_run_2)
                if features_run_2 is not None else 1.0
            ),
        )

        score = (
            self.weights["feature_importance"] * components.feature_importance_score
            + self.weights["kg_evidence"] * components.kg_evidence_score
            + self.weights["llm_confidence"] * components.llm_confidence_score
            + self.weights["fact_verification"] * components.fact_verification_score
            + self.weights["explanation_consistency"] * components.explanation_consistency_score
        )
        return ECSResult(score=round(float(score), 4), components=components)
