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
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Example CLI check (PowerShell)
Invoke-RestMethod "http://localhost:8000/search?q=Tarek%20Zia&lang=all&k=10&debug=true"
```

Environment variables:

- `BACKEND_ALLOWED_ORIGINS` – comma-separated list of origins allowed by CORS (default includes `http://localhost:3000`).

FastAPI exposes:

- `GET /health` – quick status with document count.
- `GET /meta` – dataset metadata and model info.
- `GET /search` – implements the Module B→D pipeline and returns scores in **[0,1]** along with timing data and warnings. Add `&debug=true` to inspect query variants, named entities, `_doc_id` and raw semantic scores.
- `GET /debug/index_integrity` – verifies FAISS index ↔ dataset alignment (doc IDs + fingerprint).

### Frontend

```bash
cd clir-engine/frontend
npm install
$env:NEXT_PUBLIC_API_BASE="http://localhost:8000"
npm run dev
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

### Error Distribution Workflow

1. **Export failure cases**
   ```bash
   cd clir-engine/backend
   python scripts/export_error_cases.py --mode top10 --k 10 --n 100
   # or, for single-result sampling:
   python scripts/export_error_cases.py --mode top1 --n 100
   ```
   This reads `labels_fiiled.csv` (or `labels.csv`), reruns the hybrid retriever for each query, and exports `data/processed/error_cases_template.csv`. Copy it to `error_cases_annotated.csv`, fill any missing `is_relevant` values, and annotate each failure (`is_relevant=no`) with one of the required categories: `Translation Failures`, `Named Entity Mismatch`, `Semantic Gap`, `Cross-Script Ambiguity`, `Code-Switching`. Leave `error_type` blank for relevant hits.

2. **Compute distribution + LaTeX table**
   ```bash
   cd clir-engine/backend
   python scripts/compute_error_distribution.py --denom failures
   ```
   Outputs:
   - `data/processed/error_distribution_summary.csv`
   - `data/processed/error_distribution_summary.json`
   - `data/processed/error_distribution_table.tex` (ready for the report)

This satisfies the assignment’s “Error Analysis / Error Distribution” requirement.

### Index maintenance & debugging

- After changing the semantic model or modifying corpus order, delete `backend/storage/doc_emb.npy`, `backend/storage/faiss.index`, and `backend/storage/doc_fingerprint.json`, then rebuild:
  ```bash
  cd clir-engine/backend
  python scripts/build_indexes.py
  ```
- Use `GET /debug/index_integrity` to verify FAISS ↔ dataset alignment (doc IDs and fingerprint).
- Append `&debug=true` to `/search` requests when you need translation errors, query variants, internal doc IDs, and raw semantic scores for troubleshooting.

### Quick rebuild / run checklist

```powershell
cd E:\dm_clir\clir-engine\backend
python -m venv .venv               # first time only
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: convert JSON arrays -> JSONL
python scripts\convert_json_to_jsonl.py ..\..\data\english_news.json data\processed\english_news.jsonl
python scripts\convert_json_to_jsonl.py ..\..\data\bangla_news.json  data\processed\bangla_news.jsonl

# Rebuild embeddings whenever data/config changes
Remove-Item storage\doc_emb.npy -ErrorAction SilentlyContinue
Remove-Item storage\faiss.index -ErrorAction SilentlyContinue
Remove-Item storage\doc_fingerprint.json -ErrorAction SilentlyContinue
python scripts\build_indexes.py

$env:BACKEND_ALLOWED_ORIGINS="http://localhost:3000"
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Sanity check
Invoke-RestMethod "http://localhost:8000/search?q=Tarek%20Zia&lang=all&k=10&debug=true"
```
