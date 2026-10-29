# Research Methodology

## Problem formulation
Product ranking is framed as a learning-to-rank problem: given a user u and a set of
candidate products {p_1, ..., p_n}, learn a scoring function f(u, p) such that sorting
by f approximates the ideal relevance ordering (here, operationalized as purchase
behavior: label = 1 if u purchased p, 0 otherwise).

## Hybrid modeling rationale
Three model families with complementary inductive biases are combined:
- **Gradient-boosted trees** (LightGBM, XGBoost): strong on tabular, heterogeneous
  features; naturally handle non-linear thresholds and missing values; provide
  built-in feature importances and are directly SHAP-explainable via TreeExplainer.
- **Neural ranker (MLP + pairwise hinge loss)**: can learn smoother, higher-capacity
  decision boundaries and benefits from dense embedding-style features (e.g. review
  text embeddings, graph embeddings) that trees handle less naturally.
- **DeepFM**: explicitly models second-order feature interactions (e.g. category x
  price) in closed form via its factorization-machine component, complementing both
  trees (many splits needed to approximate a cross) and a plain MLP (must learn
  crosses implicitly from data).

Combining them (default: weighted ensemble of z-normalized scores) is expected to
outperform any single family, which the ablation study in
`experiments/run_experiments.py` tests directly via paired t-tests on NDCG@10.

## Knowledge-graph rationale
Collaborative-filtering-style signals (purchase/view history) are complemented with
explicit relational structure -- category/brand membership, co-purchase, review-based
"liked" edges -- providing (a) additional ranking features (graph degree, PageRank,
learned graph embeddings) and (b) *structural evidence* for explanations (a short
graph path is intuitive, verifiable evidence a human can inspect, unlike an opaque
feature weight).

## Explainability methodology
SHAP (Shapley Additive Explanations) is used because it has a strong theoretical
grounding (the unique additive attribution method satisfying local accuracy,
missingness, and consistency) and supports both global (aggregate) and local
(per-decision) explanations. The RankSHAP-style pairwise comparison
(`explain_pairwise_ranking`) extends this to the ranking-specific question "why did
A outrank B", which plain SHAP does not answer directly.

## Hallucination mitigation methodology
Two complementary layers: (1) prompt-level grounding -- the LLM is explicitly
instructed to use only the supplied evidence and to flag sparse evidence rather than
speculate; (2) post-hoc verification -- `HallucinationDetector` checks the generated
text's entities/claims against the evidence vocabulary and computes a confidence
score, with a deterministic evidence-only fallback used whenever verification fails.
This two-layer design (prevent, then verify) is standard practice for grounding LLM
outputs in retrieval-augmented generation (RAG) systems generally.

## Evaluation methodology
Ranking quality: Precision/Recall/F1@K, MAP, MRR, NDCG@K (relevance quality),
Coverage/Novelty/Diversity (catalog-level recommendation health), ROC-AUC
(discriminative power). Explanation quality: Fidelity/Faithfulness (does the
explanation reflect what the model actually does), Stability/Consistency (is the
explanation robust), plus BLEU/ROUGE/BERTScore against reference explanations where
available. Statistical rigor: paired t-tests comparing the full hybrid ensemble
against each ablated variant and against baselines, at alpha = 0.05.

## Threats to validity
- Synthetic data (default) encodes a simplified category-affinity generative process;
  results on synthetic data should be treated as a pipeline-correctness demonstration,
  not a claim about real-world ranking quality -- re-run all experiments on a real
  dataset (Amazon/RetailRocket/MovieLens/Yelp) before drawing research conclusions.
- The lexical-overlap hallucination detector is a lighter-weight proxy for a full
  NLI-based fact verifier (documented trade-off in `llm/hallucination_detector.py`);
  it will have lower recall on heavily paraphrased hallucinations.
