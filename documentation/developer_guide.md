# Developer Guide

## Code style
- PEP-8, Python type hints on all public function signatures, docstrings on every
  module/class/function (Google-style).
- Logging via `common.config.get_logger(__name__)` -- never `print()`.
- No hardcoded hyperparameters/paths: everything reads from `configs/config.yaml`
  via `common.config.get_config()`.

## Adding a new ranking model
1. Implement a class in `ranking/models.py` with `fit(X, y, group_ids)`,
   `predict(X)`, and `feature_importance()` methods (mirror `LightGBMRanker`).
2. Register it in `ranking/ensemble.py`'s `MODEL_REGISTRY`.
3. Add its config block under `ranking.models.<name>` in `configs/config.yaml`.
4. Add a unit test in `tests/test_ranking_models.py`.

## Adding a new dataset
1. Add a `_load_<dataset>()` method to `preprocessing/data_loader.py` that maps the
   dataset's native columns onto the common schema (`users`, `products`,
   `interactions`, `reviews` -- see existing MovieLens/Yelp/RetailRocket methods for
   the pattern).
2. Add the dataset name to `SUPPORTED_DATASETS`.
3. No other module needs to change -- feature engineering, KG building, and ranking
   all operate on the common schema.

## Adding a new explainability metric
Add a function to `evaluation/metrics.py` following the existing signature pattern
(`(predictions_or_labels, ...) -> float`), then wire it into
`evaluation/evaluate.py::_ranking_report` or a new report section as appropriate.

## Extending the LLM layer
- New prompt types: add a `build_<name>_prompt()` function to `llm/prompt_templates.py`.
- New provider: add a `_generate_<provider>()` method to `llm/llm_client.py` and a
  branch in `LLMClient.generate()`.

## Running tests
```bash
pytest tests/ -v --tb=short
```
Tests are organized by pipeline stage: `test_data_pipeline.py`,
`test_feature_engineering.py`, `test_ranking_models.py`, `test_explainability.py`,
`test_metrics.py`, `test_evaluation.py`, `test_api.py`. `conftest.py` provides a
shared small synthetic dataset fixture so tests run in seconds.

## Dependency injection / separation of concerns
- `backend/routers/*.py` contain ONLY request/response handling -- no business logic.
- `backend/service.py::RankingService` is the single business-logic layer, injected
  into routers via FastAPI's `Depends(get_ranking_service)`.
- `database/db.py::get_session` is similarly injected as a FastAPI dependency,
  keeping routers decoupled from session lifecycle management.

## Exception handling
- `RankingService` raises clear `RuntimeError`/`ValueError` with actionable messages;
  routers translate these into appropriate HTTP status codes (404 for not-found,
  503-equivalent guidance for "model not trained yet").
- The LLM client and hallucination detector never raise on malformed/absent data --
  they degrade gracefully to template mode / a "low confidence" result, since an
  explanation-generation failure should never crash a ranking request.
