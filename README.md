# **CLIR Engine**

**Cross-Lingual Information Retrieval System**

This repository contains an end-to-end cross-lingual information retrieval (CLIR) stack. It features a **FastAPI** backend for semantic indexing and retrieval, and a **Next.js** frontend for the search interface. The system is designed to process and search across English and Bangla news datasets.

## **📂 Project Structure**

* **clir-engine/**: Source code for the application.  
  * **backend/**: FastAPI server, FAISS indexing, semantic models, and evaluation scripts.  
  * **frontend/**: Next.js web interface.  
* **data/**: Raw datasets (e.g., english\_news.json, bangla\_news.json).  
* **notebooks/**: Jupyter notebooks for data exploration and query pipeline experiments.
## **✅ Prerequisites**

* **Python**: 3.10+  
* **Node.js**: 18+  
* **Data**: The system expects processed data in clir-engine/backend/data/processed/.

## **🚀 Backend Setup**

The backend manages the search logic, embeddings (Sentence Transformers), and the FAISS vector index.
### **1\. Installation**

Navigate to the backend directory and set up the environment:

cd clir-engine/backend

\# Create virtual environment  
python \-m venv .venv

\# Activate virtual environment  
\# Windows (PowerShell):  
. .venv/Scripts/activate  
\# Unix/macOS:  
\# source .venv/bin/activate

\# Install dependencies  
pip install \-r requirements.txt
### **2\. Data Preparation**

If your source data is in JSON array format (located in the root data/ folder), convert it to JSONL (JSON Lines) and place it in the backend's processed directory.

\# Ensure you are inside clir-engine/backend/

\# Convert English Data  
python scripts/convert\_json\_to\_jsonl.py ../../data/english\_news.json data/processed/english\_news.jsonl

\# Convert Bangla Data  
python scripts/convert\_json\_to\_jsonl.py ../../data/bangla\_news.json data/processed/bangla\_news.jsonl
### **3\. Build Indexes**

Generate the embeddings and FAISS index. This must be run once before starting the server.

python scripts/build\_indexes.py

*Creates storage/doc\_emb.npy, storage/faiss.index, and storage/doc\_fingerprint.json.*
### **4\. Run the Server**

Start the API server.

**Windows (PowerShell):**

$env:BACKEND\_ALLOWED\_ORIGINS="http://localhost:3000"  
uvicorn app.main:app \--host 0.0.0.0 \--port 8000

**Unix/macOS:**

export BACKEND\_ALLOWED\_ORIGINS=http://localhost:3000  
uvicorn app.main:app \--host 0.0.0.0 \--port 8000

**API Endpoints:**

* GET /health: System status.  
* GET /search: Main search endpoint.  
* GET /meta: Dataset metadata.  
* GET /debug/index\_integrity: Verify FAISS ↔ Dataset alignment.

## **💻 Frontend Setup**

The frontend provides the user interface for searching and viewing results.

cd clir-engine/frontend

\# Install dependencies  
npm install

\# Run Development Server  
\# Windows (PowerShell):  
$env:NEXT\_PUBLIC\_API\_BASE="http://localhost:8000"  
npm run dev

\# Unix/macOS:  
\# export NEXT\_PUBLIC\_API\_BASE="http://localhost:8000"  
\# npm run dev

Open [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000) to view the application.

## **📊 Evaluation (Module D)**

To benchmark the ranking stack (Precision@10, Recall@50, nDCG@10, MRR):

1. Ensure clir-engine/backend/data/processed/labels\_fiiled.csv exists with columns: query, doc\_url, language, relevant, annotator.  
2. Run the evaluation script:

cd clir-engine/backend  
python scripts/run\_eval.py

Results are saved to data/processed/eval\_results.csv.

## **🔍 Error Analysis Workflow**

Follow this workflow to generate the Error Distribution Report.
### **1\. Export Failure Cases**

Identify queries where the system failed to retrieve relevant documents.

cd clir-engine/backend

\# Export top 10 results for top 100 queries  
python scripts/export\_error\_cases.py \--mode top10 \--k 10 \--n 100
### **2\. Annotate Errors**

1. Locate clir-engine/backend/data/processed/error\_cases\_template.csv.  
2. Rename it to error\_cases\_annotated.csv.  
3. Fill in is\_relevant. For non-relevant results, populate error\_type with:  
   * Translation Failures  
   * Named Entity Mismatch  
   * Semantic Gap  
   * Cross-Script Ambiguity  
   * Code-Switching
