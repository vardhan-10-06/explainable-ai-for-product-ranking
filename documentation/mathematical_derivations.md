# Mathematical Derivations

## 1. Ranking objectives

**LambdaRank (LightGBM)**: optimizes NDCG directly by weighting pairwise gradients
by |delta NDCG| from swapping a pair of documents, rather than optimizing a
classification-style loss. For a pair (i, j) with i more relevant than j:

    lambda_ij = -sigma / (1 + exp(sigma * (s_i - s_j))) * |Delta NDCG_ij|

**Pairwise ranking (XGBoost, `rank:pairwise`)**: minimizes a RankNet-style pairwise
logistic loss over within-query pairs:

    L = sum_{(i,j): y_i > y_j} log(1 + exp(-(s_i - s_j)))

**NeuralRanker / DeepFM (this project's pairwise hinge loss)**:

    L = sum_{(i,j): y_i > y_j} max(0, margin - (s_i - s_j))

with margin = 1.0 (see `ranking/models.py`).

**DeepFM's factorization-machine term** models second-order feature interactions in
closed form:

    y_FM(x) = w_0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j

computed efficiently via the standard reformulation:

    sum_{i<j} <v_i, v_j> x_i x_j = 1/2 * sum_f [ (sum_i v_{i,f} x_i)^2 - sum_i (v_{i,f} x_i)^2 ]

which is exactly the `fm_term` computation in `DeepFMRanker._DeepFMModule.forward`.

## 2. Ensemble combination (weighted strategy)

Given base model scores s_1..s_M for a row, each z-score normalized:

    z_m = (s_m - mean(s_m)) / std(s_m)

the ensemble score is:

    S_ensemble = ( sum_m w_m * z_m ) / ( sum_m w_m )

with weights w_m from `configs/config.yaml -> ranking.ensemble.weights`.

## 3. Ranking metrics

**DCG@k**: `DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)`
**NDCG@k**: `NDCG@k = DCG@k / IDCG@k`, where IDCG@k is DCG@k of the ideal ordering.
**MAP**: mean of Average Precision across queries, `AP = (1/R) * sum_{k: rel_k=1} P@k`.
**MRR**: mean of `1 / rank_of_first_relevant_item` across queries.

## 4. Explainability metrics

**Fidelity**: Pearson correlation between actual model output and the SHAP
reconstruction `base_value + sum(shap_values)`.

**Faithfulness (deletion test)**: mean `|f(x) - f(x_with_top_k_features_zeroed)|`;
larger drop => explanation correctly identified influential features.

**Stability**: mean cosine similarity between the SHAP explanation vector for x and
for x + small Gaussian noise, across several perturbations -- a local Lipschitz-style
robustness measure.

**Consistency**: mean Jaccard overlap between top-feature sets produced by two
explanation runs (e.g. before/after a retrain).

## 5. The Explainability Confidence Score (ECS) -- full derivation

See `evaluation/ecs.py` for the authoritative, in-code version of this derivation
(kept in sync with the implementation). Summary:

Five components, each in [0, 1]:

- **F_i** (feature importance concentration): `sum_{f in topK} |shap_f| / sum_{f in all} |shap_f|`
- **G_i** (knowledge-graph evidence): `1 / (1 + path_length)` if a KG path exists between
  the user and product, else 0
- **L_i** (LLM confidence): the provider's own confidence signal, or a fixed prior
  (0.8) for the deterministic template engine
- **V_i** (fact verification): `HallucinationDetector.verify(...).confidence_score`
- **C_i** (explanation consistency): Jaccard overlap between top-feature sets across
  two explanation runs for the same instance

Final score:

    ECS_i = w_F * F_i + w_G * G_i + w_L * L_i + w_V * V_i + w_C * C_i,   sum(w) = 1

Default weights (`configs/config.yaml -> ecs.weights`): 0.25 / 0.25 / 0.20 / 0.20 / 0.10.

**Interpretation**: ECS in [0, 1]. Values near 1 indicate a well-grounded, stable,
verifiable explanation; values near 0 should be treated with skepticism (e.g. routed
to a human reviewer, or replaced by the deterministic fallback text in
`llm/explanation_generator.py::_safe_fallback`).

**Why these five components and this weighting**: F and G capture whether the
*evidence itself* is strong (concentrated feature importance, an existing short graph
path); L and V capture whether the *generation process* is trustworthy (the
generator's own confidence and independent fact-checking); C captures *robustness*
of the explanation to re-generation. Equal weighting across F/G and a slightly lower
weight on C reflects that evidence-strength and verification are the primary
correctness signals, while consistency is a secondary robustness check -- but all
weights are config-driven and intended to be tuned/ablated empirically for a given
deployment (see `experiments/run_experiments.py` for the ablation methodology, which
can be extended to ECS weight sensitivity analysis).
