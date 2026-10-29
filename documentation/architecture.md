# Architecture Documentation

## System overview

```
                         +--------------------+
                         |   Streamlit UI      |
                         |  (frontend/)         |
                         +----------+----------+
                                    | HTTP (JSON)
                                    v
                         +--------------------+
                         |   FastAPI backend    |
                         |  (backend/app.py)      |
                         |  auth / search / rank / |
                         |  explain / evaluate /   |
                         |  graph routers           |
                         +----------+----------+
                                    |
                calls               v                    reads/writes
        +-----------------------------------------------------+
        |               backend/service.py                      |
        |         (RankingService - business logic)              |
        +---+-----------+-----------+-----------+---------------+
            |            |            |             |
            v            v            v             v
     +------------+ +----------+ +-----------+ +---------------+
     | Preprocess-  | | Feature   | | Hybrid     | | Explainability |
     | ing pipeline | | Engineer- | | Ranking     | | + LLM layer     |
     | (data_loader,| | ing        | | Ensemble    | | (SHAP, rank_    |
     | preprocess)  | | (features.| | (LightGBM,  | | explainer, LLM  |
     |              | | py)        | | XGBoost,    | | client, halluc- |
     |              | |            | | Neural,     | | ination detect- |
     |              | |            | | DeepFM)     | | or, ECS)         |
     +------+-------+ +-----+------+ +-----+-----+ +--------+-------+
            |               |              |               |
            v               v              v               v
     +-----------------------------------------------------------+
     |          Knowledge Graph (NetworkX default / Neo4j)         |
     |  Users - Products - Categories - Brands - Reviews            |
     +-----------------------------------------------------------+
                                    |
                                    v
                         +--------------------+
                         | SQLite / PostgreSQL |
                         | (auth, logs, eval    |
                         |  history)             |
                         +--------------------+
```

## Data flow (training)

1. `data/sample_data_generator.py` (or real dataset files) -> raw CSVs
2. `preprocessing/data_loader.py` -> common-schema tables
3. `preprocessing/preprocess.py` -> cleaned, encoded, scaled tables + train/val/test split
4. `knowledge_graph/graph_builder.py` -> graph + graph-derived features (degree,
   PageRank, embeddings)
5. `feature_engineering/features.py` -> full (user, product) feature matrix
6. `ranking/train.py` -> trains `ranking/ensemble.py`'s `HybridRankingEnsemble`
   (LightGBM + XGBoost + NeuralRanker [+ DeepFM]) and pickles it to `results/`
7. `evaluation/evaluate.py` / `experiments/run_experiments.py` -> metrics + ablation
   reports written to `results/`

## Data flow (serving)

1. `backend/service.py::RankingService.ensure_loaded()` loads the pickled ensemble,
   rebuilds the KG and feature matrix once at process startup (cached).
2. `/rank` -> scores candidate products for a user via the ensemble.
3. `/explain` -> builds an `ExplanationPackage` (SHAP local explanation + KG path),
   generates a natural-language explanation via `llm/explanation_generator.py`,
   verifies it with `llm/hallucination_detector.py`, and computes the ECS score.
4. `/graph/path` -> exposes the raw KG path for dashboard visualization.
5. `/evaluation` -> serves the latest `results/evaluation_report.json`.

## Class diagram (textual)

```
HybridRankingEnsemble
 +- models: Dict[str, BaseRanker]
 +- fit(X, y, group_ids)
 +- predict(X, group_ids=None) -> np.ndarray
 +- get_feature_importances() -> DataFrame

BaseRanker (informal interface; LightGBMRanker / XGBoostRanker / NeuralRanker / DeepFMRanker)
 +- fit(X, y, group_ids) -> self
 +- predict(X) -> np.ndarray
 +- feature_importance() -> Series

KnowledgeGraphBuilder
 +- graph: nx.MultiDiGraph
 +- build(users, products, interactions, reviews) -> nx.MultiDiGraph
 +- compute_graph_features() -> DataFrame
 +- compute_graph_embeddings() -> DataFrame
 +- explain_path(user_id, product_id) -> List[Dict]

ShapRankingExplainer
 +- explain_instances(X) -> DataFrame
 +- local_explanation(x_row) -> List[Dict]
 +- explain_pairwise_ranking(x_a, x_b) -> List[Dict]

RankExplainer
 +- explain(user_id, product_id, x_row, rank_score, compare_to=None) -> ExplanationPackage

ExplanationGenerator
 +- generate_single_explanation(package) -> GeneratedExplanation
 +- generate_comparison_explanation(package, product_b) -> GeneratedExplanation

HallucinationDetector
 +- verify(explanation_text, evidence_features, evidence_graph_path) -> VerificationResult

ExplainabilityConfidenceScore
 +- compute(all_shap_values, top_k_indices, graph_path, hallucination_confidence,
            features_run_1, features_run_2=None, llm_reported_confidence=None) -> ECSResult
```

## Sequence diagram: `/explain` request (textual)

```
Client -> FastAPI(/explain): POST {user_id, product_id}
FastAPI -> RankingService.explain(user_id, product_id)
RankingService -> HybridRankingEnsemble.predict(X_row)          : rank_score
RankingService -> RankExplainer.explain(...)
  RankExplainer -> ShapRankingExplainer.local_explanation(x_row) : top_shap_features
  RankExplainer -> KnowledgeGraphBuilder.explain_path(...)        : graph_path
RankingService -> ExplanationGenerator.generate_single_explanation(package)
  ExplanationGenerator -> LLMClient.generate(prompt)               : raw_text
  ExplanationGenerator -> HallucinationDetector.verify(...)         : VerificationResult
RankingService -> ExplainabilityConfidenceScore.compute(...)        : ECSResult
FastAPI <- RankingService: {explanation_text, confidence_score, ecs_score, ...}
Client <- FastAPI: 200 OK, ExplanationResponse
```

## Deployment diagram (textual)

```
[Docker host]
  +-- container: backend  (FastAPI, port 8000)  --volumes--> ./data ./results ./database
  +-- container: frontend (Streamlit, port 8501) --env--> API_BASE_URL=http://backend:8000
  +-- container: neo4j    (Bolt 7687, Browser 7474) --volume--> neo4j_data
  (optional) container: postgres (5432) --volume--> postgres_data
```

## Trade-off notes

- **NetworkX vs Neo4j**: NetworkX needs no external service and is fully sufficient
  for research-scale graphs; Neo4j is provided for production scale and native Cypher
  tooling. Same public interface (`explain_path`), so the choice is a config flag.
- **Weighted vs stacking ensemble**: weighted-sum is the default for robustness on
  smaller datasets; stacking is available for larger datasets where a meta-learner
  can be fit without overfitting.
- **Template vs real LLM**: the template engine guarantees zero hallucination
  (it only ever repeats supplied evidence) and needs no API key, at the cost of more
  formulaic phrasing than a real LLM. Swapping in a real model is one config change.
