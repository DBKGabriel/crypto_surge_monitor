#!/usr/bin/env bash
# Entrypoint script for Render.  Render sets the PORT environment variable
# automatically on deployment; Streamlit must be told to bind to this port
# and to listen on all interfaces.  This script installs any missing
# Python dependencies (in case pip install in the build phase failed) and
# launches the Streamlit app.

set -euo pipefail

pip install --no-cache-dir -r requirements.txt

exec streamlit run streamlit_app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0
