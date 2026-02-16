# CLIR Engine

## Cross-Lingual Information Retrieval System

This repository contains an end-to-end Cross-Lingual Information Retrieval (CLIR) stack. It features a FastAPI backend for semantic indexing and retrieval, and a Next.js frontend for the search interface. The system is designed to process and search across English and Bangla news datasets.

---

## 📂 Project Structure

* **clir-engine/**: Source code for the application.

  * **backend/**: FastAPI server, FAISS indexing, semantic models, and evaluation scripts.
  * **frontend/**: Next.js web interface.
* **data/**: Raw datasets (e.g., `english_news.json`, `bangla_news.json`).
* **notebooks/**: Jupyter notebooks for data exploration and query pipeline experiments.

---

## ✅ Prerequisites

* **Python**: 3.10+
* **Node.js**: 18+
* **Data**: The system expects processed data in `clir-engine/backend/data/processed/`.

---

# 🚀 Backend Setup

The backend manages the search logic, embeddings (Sentence Transformers), and the FAISS vector index.

## 1. Installation

Navigate to the backend directory and set up the environment:

```bash
cd clir-engine/backend

# Create virtual environment
python -m venv .venv
```

### Activate Virtual Environment

**Windows (PowerShell):**

```bash
. .venv/Scripts/activate
```

**Unix/macOS:**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Data Preparation

If your source data is in JSON array format (located in the root `data/` folder), convert it to JSONL (JSON Lines) and place it in the backend's processed directory.

Ensure you are inside `clir-engine/backend/`:

```bash
# Convert English Data
python scripts/convert_json_to_jsonl.py ../../data/english_news.json data/processed/english_news.jsonl

# Convert Bangla Data
python scripts/convert_json_to_jsonl.py ../../data/bangla_news.json data/processed/bangla_news.jsonl
```

---

## 3. Build Indexes

Generate the embeddings and FAISS index. This must be run once before starting the server.

```bash
python scripts/build_indexes.py
```

This creates:

* `storage/doc_emb.npy`
* `storage/faiss.index`
* `storage/doc_fingerprint.json`

---

## 4. Run the Server

Start the API server.

### Windows (PowerShell)

```bash
$env:BACKEND_ALLOWED_ORIGINS="http://localhost:3000"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Unix/macOS

```bash
export BACKEND_ALLOWED_ORIGINS=http://localhost:3000
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

* `GET /health` — System status
* `GET /search` — Main search endpoint
* `GET /meta` — Dataset metadata
* `GET /debug/index_integrity` — Verify FAISS ↔ Dataset alignment

---

# 💻 Frontend Setup

The frontend provides the user interface for searching and viewing results.

```bash
cd clir-engine/frontend

# Install dependencies
npm install
```

### Run Development Server

**Windows (PowerShell):**

```bash
$env:NEXT_PUBLIC_API_BASE="http://localhost:8000"
npm run dev
```

**Unix/macOS:**

```bash
export NEXT_PUBLIC_API_BASE="http://localhost:8000"
npm run dev
```

Open the application at:

```
http://localhost:3000
```

---

# 📊 Evaluation (Module D)

To benchmark the ranking stack (Precision@10, Recall@50, nDCG@10, MRR):

1. Ensure `clir-engine/backend/data/processed/labels_fiiled.csv` exists with columns:

   ```
   query, doc_url, language, relevant, annotator
   ```

2. Run the evaluation script:

```bash
cd clir-engine/backend
python scripts/run_eval.py
```

Results are saved to:

```
data/processed/eval_results.csv
```

---

# 🔍 Error Analysis Workflow

Follow this workflow to generate the Error Distribution Report.

## 1. Export Failure Cases

Identify queries where the system failed to retrieve relevant documents.

```bash
cd clir-engine/backend

# Export top 10 results for top 100 queries
python scripts/export_error_cases.py --mode top10 --k 10 --n 100
```

---

## 2. Annotate Errors

1. Locate `clir-engine/backend/data/processed/error_cases_template.csv`.
2. Rename it to `error_cases_annotated.csv`.
3. Fill in `is_relevant`.
4. For non-relevant results, populate `error_type` with one of the following:

* Translation Failures
* Named Entity Mismatch
* Semantic Gap
* Cross-Script Ambiguity
* Code-Switching

---

## 3. Generate Report

Compute statistics and generate the LaTeX table.

```bash
# Inside clir-engine/backend/
python scripts/compute_error_distribution.py --denom failures
```

Outputs:

* `data/processed/error_distribution_summary.json`
* `data/processed/error_distribution_table.tex`

---

# 🛠 Maintenance & Debugging

## Rebuilding Indexes

If you modify the semantic model or the corpus data:

1. **Stop the server.**

2. **Clean old artifacts:**

   **Windows (from clir-engine/backend):**

   ```bash
   Remove-Item storage\doc_emb.npy, storage\faiss.index, storage\doc_fingerprint.json
   ```

   **Unix (from clir-engine/backend):**

   ```bash
   rm storage/doc_emb.npy storage/faiss.index storage/doc_fingerprint.json
   ```

3. **Rebuild:**

   ```bash
   python scripts/build_indexes.py
   ```

---

## Debugging Search

Append `&debug=true` to any search query URL to inspect translation steps, query variants, and internal scores.

Example:

```
http://localhost:8000/search?q=climate&debug=true
```

---

