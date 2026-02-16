# **CLIR Engine**

**Cross-Lingual Information Retrieval System**

This repository contains an end-to-end cross-lingual information retrieval (CLIR) stack. It features a **FastAPI** backend for semantic indexing and retrieval, and a **Next.js** frontend for the search interface. The system is designed to process and search across English and Bangla news datasets.

## **📂 Project Structure**

* **clir-engine/**: Source code for the application.  
  * **backend/**: FastAPI server, FAISS indexing, semantic models, and evaluation scripts.  
  * **frontend/**: Next.js web interface.  
* **data/**: Raw datasets (e.g., english\_news.json, bangla\_news.json).  
* **notebooks/**: Jupyter notebooks for data exploration and query pipeline experiments.
