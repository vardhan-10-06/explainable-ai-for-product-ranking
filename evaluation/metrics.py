"""
evaluation/metrics.py
=======================
Standard ranking, explainability, and NLP evaluation metrics.

Ranking metrics: Precision@K, Recall@K, F1@K, MAP, MRR, NDCG@K, Coverage,
                 Novelty, Diversity, ROC-AUC.
Explainability metrics: Fidelity, Faithfulness, Stability, Consistency.
NLP metrics for generated explanations: BLEU, ROUGE-L, BERTScore (with a
                 dependency-light fallback if the `bert-score` package is
                 unavailable).
Efficiency metrics: latency / inference-time / memory-usage measurement
                 helpers used by evaluation/evaluate.py.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from common.config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Ranking metrics
# =============================================================================
def precision_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """Fraction of the top-k ranked items that are relevant (label == 1)."""
    top_k = ranked_labels[:k]
    return float(np.mean(top_k)) if top_k else 0.0


def recall_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    total_relevant = sum(ranked_labels)
    if total_relevant == 0:
        return 0.0
    return float(sum(ranked_labels[:k]) / total_relevant)


def f1_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p, r = precision_at_k(ranked_labels, k), recall_at_k(ranked_labels, k)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def average_precision(ranked_labels: Sequence[int]) -> float:
    """Average Precision for a single ranked list (used to compute MAP across users)."""
    hits, sum_precisions = 0, 0.0
    for i, label in enumerate(ranked_labels, start=1):
        if label:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / hits if hits > 0 else 0.0


def mean_average_precision(list_of_ranked_labels: List[Sequence[int]]) -> float:
    """MAP across multiple users' ranked lists."""
    if not list_of_ranked_labels:
        return 0.0
    return float(np.mean([average_precision(labels) for labels in list_of_ranked_labels]))


def reciprocal_rank(ranked_labels: Sequence[int]) -> float:
    """Reciprocal rank of the first relevant item (0 if none found)."""
    for i, label in enumerate(ranked_labels, start=1):
        if label:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(list_of_ranked_labels: List[Sequence[int]]) -> float:
    """MRR across multiple users' ranked lists."""
    if not list_of_ranked_labels:
        return 0.0
    return float(np.mean([reciprocal_rank(labels) for labels in list_of_ranked_labels]))


def dcg_at_k(ranked_labels: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    labels = np.asarray(ranked_labels[:k], dtype=float)
    if labels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(labels / discounts))


def ndcg_at_k(ranked_labels: Sequence[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    ideal = sorted(ranked_labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_labels, k) / idcg


def coverage(recommended_item_sets: List[set], catalog_size: int) -> float:
    """Fraction of the full catalog that appears across all users' recommendation lists."""
    if catalog_size == 0:
        return 0.0
    all_recommended = set().union(*recommended_item_sets) if recommended_item_sets else set()
    return len(all_recommended) / catalog_size


def novelty(recommended_item_ids: List[str], item_popularity: dict) -> float:
    """
    Mean self-information novelty: -log2(popularity) averaged over
    recommended items; higher = recommending less-obvious items.
    """
    if not recommended_item_ids:
        return 0.0
    scores = []
    for item in recommended_item_ids:
        p = max(item_popularity.get(item, 1e-6), 1e-6)
        scores.append(-np.log2(p))
    return float(np.mean(scores))


def diversity(item_feature_vectors: np.ndarray) -> float:
    """
    Intra-list diversity: mean pairwise cosine distance between the
    feature vectors of recommended items (higher = more diverse list).
    """
    if len(item_feature_vectors) < 2:
        return 0.0
    normed = item_feature_vectors / (np.linalg.norm(item_feature_vectors, axis=1, keepdims=True) + 1e-8)
    sim_matrix = normed @ normed.T
    n = len(item_feature_vectors)
    off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
    mean_sim = off_diag_sum / (n * (n - 1))
    return float(1 - mean_sim)


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC; returns NaN gracefully if only one class is present."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        logger.warning("ROC-AUC undefined (only one class present in y_true); returning nan.")
        return float("nan")


# =============================================================================
# Explainability metrics
# =============================================================================
def fidelity(model_predict_fn: Callable, X, shap_values: np.ndarray, base_value: float) -> float:
    """
    Fidelity: correlation between the model's actual predictions and the
    SHAP-value-reconstructed predictions (base_value + sum(shap_values)).
    High fidelity means the explanation faithfully reconstructs the model's
    own output.
    """
    actual = model_predict_fn(X)
    reconstructed = base_value + shap_values.sum(axis=1)
    if np.std(actual) == 0 or np.std(reconstructed) == 0:
        return 0.0
    return float(np.corrcoef(actual, reconstructed)[0, 1])


def faithfulness(model_predict_fn: Callable, X, shap_values: np.ndarray, top_k: int = 3) -> float:
    """
    Faithfulness ("deletion" test): mean drop in prediction when the top-k
    most important (by |SHAP|) features are zeroed out. Larger drop implies
    the explanation correctly identified features the model actually relies on.
    """
    X_arr = np.asarray(X)
    drops = []
    for i in range(len(X_arr)):
        original_pred = model_predict_fn(X_arr[i:i + 1])[0]
        top_features = np.argsort(-np.abs(shap_values[i]))[:top_k]
        perturbed = X_arr[i].copy()
        perturbed[top_features] = 0.0
        perturbed_pred = model_predict_fn(perturbed.reshape(1, -1))[0]
        drops.append(abs(original_pred - perturbed_pred))
    return float(np.mean(drops))


def stability(explain_fn: Callable, x_row, n_perturbations: int = 5, noise_scale: float = 0.01) -> float:
    """
    Stability: how similar the explanation stays under small input
    perturbations (Lipschitz-style local robustness of explanations).
    Returns mean cosine similarity between the original and perturbed
    SHAP vectors (1.0 = perfectly stable).
    """
    rng = np.random.default_rng(42)
    original = np.asarray(explain_fn(x_row))
    sims = []
    for _ in range(n_perturbations):
        noise = rng.normal(0, noise_scale, size=x_row.shape)
        perturbed_row = x_row + noise
        perturbed_explanation = np.asarray(explain_fn(perturbed_row))
        denom = (np.linalg.norm(original) * np.linalg.norm(perturbed_explanation)) + 1e-8
        sims.append(float(np.dot(original, perturbed_explanation) / denom))
    return float(np.mean(sims))


def consistency(explanations_a: List[List[str]], explanations_b: List[List[str]]) -> float:
    """
    Consistency: mean Jaccard overlap between two sets of top-feature
    explanations for the same instances (e.g. across two model retrains),
    checking whether the *explanations* (not just predictions) are stable.
    """
    if not explanations_a or len(explanations_a) != len(explanations_b):
        return 0.0
    scores = []
    for a, b in zip(explanations_a, explanations_b):
        set_a, set_b = set(a), set(b)
        union = set_a | set_b
        scores.append(len(set_a & set_b) / len(union) if union else 1.0)
    return float(np.mean(scores))


# =============================================================================
# NLP metrics for generated explanations
# =============================================================================
def bleu_score(candidate: str, reference: str, max_n: int = 4) -> float:
    """
    Lightweight BLEU-N implementation (no external dependency) with
    a brevity penalty, for scoring generated explanations against a
    reference explanation.
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = _ngrams(cand_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(cand_ngrams[g], ref_ngrams.get(g, 0)) for g in cand_ngrams)
        precisions.append(overlap / sum(cand_ngrams.values()))

    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        geo_mean = float(np.exp(np.mean(np.log(precisions))))

    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return float(bp * geo_mean)


def _ngrams(tokens: List[str], n: int) -> dict:
    counts: dict = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L (longest common subsequence based F-measure)."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    lcs_len = _lcs_length(cand_tokens, ref_tokens)
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    return 2 * precision * recall / (precision + recall + 1e-8)


def _lcs_length(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def bert_score(candidate: str, reference: str) -> float:
    """
    BERTScore F1 between a candidate and reference explanation. Falls back
    to a TF-IDF cosine-similarity proxy if the `bert-score` package (and
    its underlying transformer download) is unavailable, so evaluation
    still runs in offline/CI environments.
    """
    try:
        from bert_score import score as _bertscore  # optional heavy dependency
        _, _, f1 = _bertscore([candidate], [reference], lang="en", verbose=False)
        return float(f1.mean())
    except Exception as exc:  # pragma: no cover
        logger.warning("bert-score unavailable (%s); using TF-IDF cosine similarity fallback.", exc)
        return _tfidf_cosine_fallback(candidate, reference)


def _tfidf_cosine_fallback(candidate: str, reference: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not candidate.strip() or not reference.strip():
        return 0.0
    vectorizer = TfidfVectorizer().fit([candidate, reference])
    vectors = vectorizer.transform([candidate, reference])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


# =============================================================================
# Efficiency metrics
# =============================================================================
def measure_latency(fn: Callable, *args, **kwargs) -> float:
    """Measure wall-clock latency (seconds) of a single call to `fn`."""
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def measure_memory_usage(fn: Callable, *args, **kwargs) -> float:
    """Measure peak memory usage (MB) during a call to `fn`."""
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)
