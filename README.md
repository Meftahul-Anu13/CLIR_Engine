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


