# =============================================================================
# Dockerfile - Explainable AI Product Ranking backend + ML pipeline
#
# GPU support: this base image is CPU-only for portability. For GPU
# training/inference, swap the base image for a CUDA-enabled PyTorch image,
# e.g.:
#   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# and remove the CPU-only torch install line below (pip will pull the CUDA
# build instead). The application code requires no changes either way --
# ranking/models.py auto-detects CUDA via `torch.cuda.is_available()`.
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed results database

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
