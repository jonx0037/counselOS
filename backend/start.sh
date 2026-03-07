#!/usr/bin/env bash
set -e

echo "Seeding RAG knowledge base..."
python -m rag.seed_data

echo "Starting CounselOS API..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
