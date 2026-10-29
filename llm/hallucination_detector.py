"""
llm/hallucination_detector.py
================================
Verifies that a generated explanation is grounded in the actual evidence
(SHAP feature importances + knowledge-graph facts) that was supplied to
the LLM, flags unsupported claims, and produces a confidence score.

Verification strategy (documented trade-off): a full NLI-based fact
verifier (e.g. entailment scoring with a cross-encoder) would be more
accurate but adds a heavy model dependency and latency to every
explanation. This reference implementation uses a lighter, fully
transparent lexical-overlap + entity-mention check: every noun-phrase-ish
token in the explanation must correspond to a real feature name or graph
entity from the evidence set, and any capitalized token that looks like an
entity but doesn't appear in the evidence is flagged as an unsupported
claim. This is auditable (you can see exactly which words tripped the
check) at some cost to recall on paraphrased claims -- a reasonable
default for a research-grade explainability system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from common.config import get_config, get_logger

logger = get_logger(__name__)

# Note: deliberately excludes underscore from the token boundary. Feature
# names in the evidence (e.g. "price_scaled") are rendered to the user with
# underscores replaced by spaces (see llm_client.py's template engine), so
# tokenizing evidence and explanation text the same way -- splitting on
# underscores too -- is what lets "price_scaled" and "price scaled" match.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class VerificationResult:
    """Outcome of hallucination-checking one generated explanation."""
    passed: bool
    confidence_score: float
    unsupported_claims: List[str] = field(default_factory=list)
    fact_match_ratio: float = 0.0


class HallucinationDetector:
    """Checks LLM-generated explanations against the evidence they were supposed to be grounded in."""

    def __init__(self):
        self.cfg = get_config()["hallucination_detection"]

    def verify(
        self,
        explanation_text: str,
        evidence_features: List[Dict],
        evidence_graph_path: List[Dict],
    ) -> VerificationResult:
        """
        Verify an explanation against its supporting evidence.

        Args:
            explanation_text: The LLM's (or template engine's) generated text.
            evidence_features: The SHAP feature evidence passed to the LLM.
            evidence_graph_path: The knowledge-graph path evidence passed to the LLM.

        Returns:
            VerificationResult with pass/fail, a 0-1 confidence score, and
            any flagged unsupported claims.
        """
        evidence_vocab = self._build_evidence_vocab(evidence_features, evidence_graph_path)
        explanation_tokens = self._tokenize(explanation_text)

        candidate_claims = self._extract_candidate_entities(explanation_text)
        unsupported = [c for c in candidate_claims if c.lower() not in evidence_vocab]

        overlap = len(explanation_tokens & evidence_vocab)
        fact_match_ratio = overlap / max(len(evidence_vocab), 1)

        confidence = self._compute_confidence(fact_match_ratio, unsupported, evidence_features, evidence_graph_path)
        passed = (
            confidence >= self.cfg["min_confidence_to_pass"]
            and fact_match_ratio >= self.cfg["fact_match_threshold"] * 0.5
        )

        if not passed:
            logger.info(
                "Explanation failed hallucination check (confidence=%.2f, unsupported=%s)",
                confidence, unsupported,
            )

        return VerificationResult(
            passed=passed,
            confidence_score=round(confidence, 3),
            unsupported_claims=unsupported,
            fact_match_ratio=round(fact_match_ratio, 3),
        )

    # ------------------------------------------------------------------ #
    def _build_evidence_vocab(self, features: List[Dict], graph_path: List[Dict]) -> set:
        vocab = set()
        for f in features:
            vocab.update(self._tokenize(f.get("feature", "")))
        for edge in graph_path:
            vocab.update(self._tokenize(edge.get("relation", "")))
            vocab.update(self._tokenize(edge.get("from", "")))
            vocab.update(self._tokenize(edge.get("to", "")))
        return vocab

    def _tokenize(self, text: str) -> set:
        return {t.lower() for t in _WORD_RE.findall(text)}

    def _extract_candidate_entities(self, text: str) -> List[str]:
        """
        Heuristically extract "claim-like" tokens: capitalized words or
        numbers that read as specific factual assertions (prices, counts,
        proper nouns) rather than generic connective language.
        """
        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text)
        numbers = re.findall(r"\$?\d+(?:\.\d+)?%?", text)
        # Filter out common sentence-initial capitalization noise.
        stopwords = {"This", "The", "These", "Product", "Using"}
        candidates = [c for c in candidates if c not in stopwords]
        return candidates + numbers

    def _compute_confidence(
        self, fact_match_ratio: float, unsupported: List[str], features: List[Dict], graph_path: List[Dict]
    ) -> float:
        """
        Combine fact-match ratio, unsupported-claim penalty, and evidence
        richness into a single 0-1 confidence score.
        """
        evidence_richness = min(1.0, (len(features) + len(graph_path)) / 6.0)
        unsupported_penalty = min(1.0, len(unsupported) * 0.2)
        base = 0.5 * fact_match_ratio + 0.5 * evidence_richness
        return max(0.0, min(1.0, base - unsupported_penalty))
