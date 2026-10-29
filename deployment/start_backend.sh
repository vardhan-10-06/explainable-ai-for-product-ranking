#!/bin/sh
set -eu

export CONFIG_PATH="${CONFIG_PATH:-/app/deployment/render-config.yaml}"
data_root="${RANKWISE_DATA_DIR:-/var/data}"
raw_dir="$data_root/raw"
results_dir="$data_root/results"
ensemble_path="$results_dir/hybrid_ensemble.pkl"

mkdir -p "$raw_dir" "$results_dir"
python -m deployment.prepare_render_data

if [ ! -s "$ensemble_path" ]; then
    python -m ranking.train --output "$ensemble_path"
fi

mkdir -p /app/results
ln -sfn "$ensemble_path" /app/results/hybrid_ensemble.pkl

exec python -m uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port "${PORT:-10000}" \
    --proxy-headers
