## CLIR Engine

End-to-end cross-lingual information retrieval stack with FastAPI (backend) and Next.js (frontend). Uses the processed news datasets stored under `backend/data/processed/`.

### Prerequisites

- Python 3.10+
- Node 18+
- The processed datasets (`english_news.jsonl`, `bangla_news.jsonl`) already live in `backend/data/processed/`.

### Backend

```bash
cd clir-engine/backend
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt

# (Optional) convert JSON arrays -> JSONL if you only have *.json
python scripts/convert_json_to_jsonl.py ../../data/english_news.json data/processed/english_news.jsonl
python scripts/convert_json_to_jsonl.py ../../data/bangla_news.json data/processed/bangla_news.jsonl

# Build sentence-transformer embeddings + FAISS index once
python scripts/build_indexes.py

# Run API
export BACKEND_ALLOWED_ORIGINS=http://localhost:3000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Environment variables:

- `BACKEND_ALLOWED_ORIGINS` – comma-separated list of origins allowed by CORS (default includes `http://localhost:3000`).

FastAPI exposes:

- `GET /health` – quick status with document count.
- `GET /meta` – dataset metadata and model info.
- `GET /search` – implements the Module B→D pipeline and returns scores in **[0,1]** along with timing data and warnings.

### Frontend

```bash
cd clir-engine/frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Env vars:

- `NEXT_PUBLIC_API_BASE` – backend base URL (defaults to `http://localhost:8000` if unset).

The UI fetches `/meta` on load and `/search?q=...&lang=...&k=...` on demand, rendering warnings, timing breakdown, results, and optional debug scores.

### Evaluation (Module D)

Populate `backend/data/processed/labels_fiiled.csv` (or `labels.csv`) with columns `query,doc_url,language,relevant,annotator`. Then:

```bash
cd clir-engine/backend
python scripts/run_eval.py
```

The script runs the in-process ranking stack for each labeled query, computes Precision@10, Recall@50, nDCG@10, MRR, saves `data/processed/eval_results.csv`, and prints overall averages.
