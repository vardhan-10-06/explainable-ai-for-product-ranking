#!/bin/sh
set -eu

exec python -m streamlit run frontend/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-10000}" \
    --server.headless true
