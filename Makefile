.PHONY: install seed backend frontend test lint

install:
	cd backend && pip install -e ".[dev]"
	cd frontend && npm install

seed:
	cd backend && python -m rag.seed_data

backend:
	cd backend && uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && pytest -v

lint:
	cd backend && ruff check . && mypy .
	cd frontend && npm run type-check

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf backend/data/chroma
