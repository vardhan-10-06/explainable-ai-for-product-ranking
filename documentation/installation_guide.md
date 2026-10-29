# Installation Guide

## Prerequisites
- Python 3.10+ (3.11 recommended)
- Docker + docker-compose (optional, for containerized deployment)
- A Neo4j instance (optional -- only required if `knowledge_graph.backend: neo4j`)
- ~2GB free disk for model artifacts and dataset cache

## Local (bare-metal) installation

```bash
git clone <repo-url> && cd product_ranking_xai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### GPU support
PyTorch auto-detects CUDA (`torch.cuda.is_available()`); no code changes are needed.
Install a CUDA-enabled PyTorch build if you have a GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
If no GPU is available, the neural ranker and DeepFM model fall back to CPU
automatically.

## Generate data and train

```bash
python -m data.sample_data_generator --output-dir data/raw --n-users 5000 --n-products 1000
python -m ranking.train --tune          # --tune runs hyperparameter search first
python -m evaluation.evaluate
python -m experiments.run_experiments
```

## Run the API and dashboard

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
# in a second terminal:
API_BASE_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Register a user and get a token before using the dashboard/API:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "demo1234", "role": "admin"}'
```

## Docker installation

```bash
cp .env.example .env       # set JWT_SECRET_KEY (and LLM_API_KEY if using a real LLM)
docker-compose up --build
```
This starts the backend (port 8000), the Streamlit frontend (port 8501), and a Neo4j
instance (ports 7474/7687). The backend container automatically generates a synthetic
dataset and trains the ensemble on first startup.

## Using a real LLM provider

Set in `configs/config.yaml`:
```yaml
llm:
  provider: "openai_compatible"
  openai_compatible:
    base_url: "https://api.openai.com/v1"    # or any OpenAI-compatible endpoint
    model: "gpt-4o-mini"                       # or a self-hosted Llama/Qwen model name
    api_key_env: "LLM_API_KEY"
```
and export `LLM_API_KEY` in your environment (or `.env` for Docker). Self-hosted Llama/
Qwen deployments (vLLM, TGI, Ollama's OpenAI shim) work identically -- only `base_url`
and `model` change.

## Switching to Neo4j

1. Start a Neo4j instance (`docker-compose up neo4j` or a managed instance).
2. Stage the CSVs referenced in `knowledge_graph/cypher_queries.cql` under Neo4j's
   `import/` directory.
3. Set `knowledge_graph.backend: neo4j` and fill in `knowledge_graph.neo4j.*` in
   `configs/config.yaml`.
4. Run the setup script: `python -c "from knowledge_graph.neo4j_adapter import Neo4jGraphAdapter; Neo4jGraphAdapter().run_setup_script()"`

## Switching to PostgreSQL

Set `database.url` in `configs/config.yaml` to
`postgresql://user:password@host:5432/dbname` and `pip install psycopg2-binary`.
