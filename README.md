# Explainable AI for Product Ranking Using Hybrid Machine Learning, Knowledge Graphs, and LLMs

A research-grade, modular reference implementation of an explainable product-ranking
system that combines:

- **Hybrid ranking**: LightGBM (LambdaRank) + XGBoost (pairwise) + a PyTorch neural
  ranker + a DeepFM-style model, combined via a configurable ensemble strategy.
- **Knowledge graph reasoning**: users, products, categories, brands, and reviews
  connected by `PURCHASED`, `VIEWED`, `BELONGS_TO`, `SIMILAR_TO`, `LIKED`, `REVIEWED`,
  and `BOUGHT_WITH` relationships (NetworkX by default, Neo4j-ready via Cypher scripts).
- **Explainability**: SHAP local/global explanations plus a RankSHAP-style pairwise
  "why A over B" comparison.
- **LLM-generated, hallucination-checked explanations**: natural-language explanations
  grounded strictly in SHAP + knowledge-graph evidence, verified before being shown
  to a user, with a novel **Explainability Confidence Score (ECS)**.
- **Full evaluation suite**: ranking metrics (Precision/Recall/F1@K, MAP, MRR, NDCG,
  Coverage, Novelty, Diversity, ROC-AUC), explainability metrics (Fidelity,
  Faithfulness, Stability, Consistency), and NLP metrics for the generated text
  (BLEU, ROUGE-L, BERTScore).
- **FastAPI backend + Streamlit dashboard**, Docker deployment, SQLite/PostgreSQL/Neo4j
  database support, and a full pytest suite.

## Honest scope note

This is a complete, runnable **reference implementation**, not a claim that every one
of the 20 items in the original brief exists as a polished, standalone artifact. In
particular:
- The "IEEE paper draft" and "presentation slides" ship as detailed Markdown
  (`documentation/ieee_paper_draft.md`, `documentation/presentation_outline.md`)
  rather than a compiled PDF/PPTX.
- The knowledge graph defaults to an in-memory NetworkX backend so the whole project
  runs with zero external services; Neo4j is fully supported (see
  `knowledge_graph/cypher_queries.cql` and `knowledge_graph/neo4j_adapter.py`) as a
  one-line config change once you have a Neo4j instance running.
- The dataset loaders are schema-normalizing adapters (see "Datasets" below) rather
  than four independently hand-tuned pipelines -- this is the standard way production
  systems ingest heterogeneous sources, and it's documented as a deliberate trade-off
  in `preprocessing/data_loader.py`.
- GPU support is available (PyTorch auto-detects CUDA) but was developed/tested on CPU
  in this environment; see the Dockerfile comment for the one-line GPU base-image swap.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Put flipkart_combined.csv in data/raw/ (see "Datasets" below)

# 2. Train the hybrid ensemble
python -m ranking.train                 # add --tune for hyperparameter search

# 3. Evaluate
python -m evaluation.evaluate
python -m evaluation.evaluate_datasets  # per-dataset size, ranking metrics, and efficiency
python -m experiments.run_experiments   # ablation study + significance testing

# 4. Serve the API
uvicorn backend.app:app --reload

# 5. Launch the dashboard (in a separate terminal)
export API_BASE_URL=http://localhost:8000
streamlit run frontend/streamlit_app.py
```

The ranking API scores the user's eligible product catalog, including products
they have not interacted with, and omits products with a recorded purchase when
no explicit candidate list is supplied. Model training excludes the purchase
indicator because it is also the relevance label; retrain existing model
artifacts with `python -m ranking.train` before starting the API.

Or with Docker:
```bash
cp .env.example .env   # fill in a real JWT_SECRET_KEY
docker-compose up --build
```

### Run from VS Code

Open the repository root folder in VS Code, select **Rankwise: Run app** in
**Run and Debug**, then press **F5** (or **Ctrl+F5** to run without debugging).
This starts the API and dashboard together, prints
`Dashboard URL: http://127.0.0.1:8501` in the terminal, and opens the dashboard.
Alternatively, run the **Rankwise: Run app** task from **Terminal > Run Task**.

## Datasets

Set `data.dataset` in `configs/config.yaml` to one of `amazon`, `flipkart`,
`retailrocket`, `movielens`, or `yelp`. Every dataset is normalized to a common schema
(`users`, `products`, `interactions`, `reviews`) by `preprocessing/data_loader.py`:

| Dataset      | Expected raw files under `data/raw/`                                   |
|--------------|--------------------------------------------------------------------------|
| amazon       | `users.csv`, `products.csv`, `interactions.csv`, `reviews.csv` (the synthetic generator produces this format directly) |
| flipkart     | `flipkart_combined.csv` (product ratings become binary relevance labels at 4 stars; product category is the ranking group because the source has no customer IDs) |
| retailrocket | `events.csv` (official RetailRocket export)                             |
| movielens    | `ratings.csv`, `movies.csv` (official MovieLens export)                 |
| yelp         | `yelp_academic_dataset_review.json`, `yelp_academic_dataset_business.json` |

The current configuration is set to Flipkart. Copy the supplied CSV to
`data/raw/flipkart_combined.csv`, then run the training and evaluation commands
above. The train/validation/test split is 70/15/15 within each product category.
Products without a rating are retained in the catalog where applicable but are
not treated as negative training examples (they are excluded from the labeled
train/validation/test rows). Flipkart ranking features include product title and
key-feature text, category and brand, and review/rating counts. Listing price and
average rating are excluded from model inputs: the supplied rating is used only
to create the relevance label (4 stars or higher), avoiding direct target leakage.
Price remains visible in the catalog and ranking results but does not affect rank.

If the dataset-specific files aren't found, the loader falls back to the common-schema
CSVs so the pipeline still runs against synthetic data.

### Compare local datasets

Run `python -m evaluation.evaluate_datasets` to train and evaluate each locally
available supported dataset independently. The report is written to
`results/dataset_comparison.json` and displayed on the dashboard's Evaluation page.
It includes source file sizes, split sizes, ranking metrics, inference latency and
memory, and train/evaluation time. CSV line counts are physical newline counts, not
necessarily logical records when quoted fields contain embedded newlines. The full
normalized Amazon Electronics import is inventoried but not locally trained by this
command because of its size; the runtime slice is evaluated instead. Datasets whose
source files are absent are reported as not present and have no metrics.

Choose **Laptops**, **Mobiles**, or **Earphones** from the Discover category
dropdown to carry the selection into Rankings, Explainability, and category-specific
held-out metrics in Evaluation. The separate free-text search accepts product names,
brands, or other text and searches only within the selected category. Changing the
Discover category clears any ranking, search result, or explanation from the previous
category. The overall dataset comparison remains available alongside the selected
category's Flipkart metrics. Search results are used as the candidate set for the
next ranking; when all results have one brand, Evaluation shows held-out metrics for
that category/brand. Evaluation includes hybrid-vs-popularity metric charts alongside
the existing metrics and dataset comparison tables.

The popularity baseline is computed from training-split interaction counts only;
unseen test products receive a zero score. Ranking metrics (Precision/Recall/F1@K,
NDCG, MAP, MRR, coverage, and novelty) average over possible orderings of equal-score
items, avoiding arbitrary advantages from CSV row order. ROC-AUC also treats tied
scores evenly.

### UCSD Amazon Electronics import

The project includes an import path for the UCSD/McAuley Amazon Electronics
5-core review data. The original compressed files are preserved under
`data/raw/ucsd_electronics/`. The complete normalized import is under
`data/raw/ucsd_electronics_normalized/`, while the active local runtime uses
the smaller deterministic slice under `data/raw/ucsd_electronics_runtime/`.
The runtime slice currently contains 5,000 users, 21,419 products before
preprocessing filters, and 43,939 interactions/reviews. Ratings 4-5 are
positive purchase-like signals and ratings 1-3 are negative view signals.

## Project structure

```
data/                 dataset generator + raw/processed storage
preprocessing/         loading, cleaning, encoding, splitting
feature_engineering/   user/product/interaction/temporal/graph features
ranking/               LightGBM, XGBoost, Neural, DeepFM models + hybrid ensemble
knowledge_graph/       NetworkX graph builder, Cypher scripts, Neo4j adapter
explainability/        SHAP explainer, rank explainer (SHAP + KG evidence)
llm/                    prompt templates, LLM client, explanation generator,
                        hallucination detector
evaluation/             ranking/explainability/NLP/efficiency metrics, ECS metric,
                        evaluation report generator
experiments/            baseline/ablation/significance-testing runner
backend/                FastAPI app, routers, auth, business-logic service layer
database/               SQLAlchemy models, schema.sql, engine/session management
frontend/               Streamlit dashboard
configs/                central YAML configuration
documentation/          full doc set (see below)
tests/                  pytest unit/integration/API tests
```

## Documentation

See `documentation/` for the installation guide, developer guide, user manual,
architecture documentation (with diagrams), API reference, research methodology,
mathematical derivations (including the ECS formulation), the IEEE paper draft, and
the presentation outline.

## Testing

```bash
pytest tests/ -v
```

## The Explainability Confidence Score (ECS)

The project's novel research contribution -- a single 0-1 score combining feature
importance concentration, knowledge-graph evidence strength, LLM confidence, fact
verification, and explanation consistency. Full mathematical formulation and
rationale: `evaluation/ecs.py` docstring and `documentation/mathematical_derivations.md`.
