# Biblios 📚
### Production RAG & AI Evaluation Harness

**Biblios** is a high-performance semantic discovery engine and retrieval-augmented generation (RAG) backend. By transitioning probabilistic models into deterministic, fault-tolerant tools, it executes low-latency contextual queries over multi-dimensional text metadata while maintaining rigorous quality control through automated LLM-as-a-judge evaluation.

---

## 🛠️ The Tech Stack

* **Language:** Python 3.13
* **API Architecture:** FastAPI, Uvicorn, Pydantic (Strict JSON Schema Validation)
* **AI/LLM Frameworks:** LangChain, Hugging Face Transformers
* **Evaluation LLM:** Anthropic API (Claude 4.5 Haiku)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local Execution)
* **Vector Database:** ChromaDB (via `langchain-chroma`)
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
* **Secrets Management:** `python-dotenv`

---

## 🚀 Production Architecture

The project is structured to manage the full lifecycle of Applied AI systems, from data ingestion to automated production observability:

### **01. Data Preprocessing & Telemetry**
* **Exploratory Data Analysis (EDA):** Performed correlation analysis using Spearman heatmaps to ensure data missingness was not biased.
* **Data Cleaning:** Engineered feature pipelines and filtered for semantic density (25-word minimum description threshold) to optimize inference.
* **Tagging:** Prepended unique ISBN identifiers to descriptions to allow for precise metadata retrieval and downstream system integration.

### **02. Vector Search (Offline-Tolerant Infrastructure)**
* **Semantic Embeddings:** Swapped cloud-dependent embeddings for a locally-hosted Hugging Face model (`all-MiniLM-L6-v2`) to generate high-dimensional vectors for 5,000+ records, prioritizing reliability and reduced latency.
* **Indexing:** Utilized **ChromaDB** for efficient K-Nearest Neighbor (KNN) retrieval.
* **Document Isolation:** Bypassed arbitrary character-chunking in favor of strict, per-document splitting to prevent vector dilution and ensure high-fidelity retrieval.

### **03. Machine Learning Pipelines**
* **Zero-Shot Classification:** Implemented `facebook/bart-large-mnli` to autonomously classify books into broad categories (Fiction vs. Non-Fiction).
* **Sentiment Analysis:** Leveraged a fine-tuned **RoBERTa** model to detect sentence-level sentiment across seven discrete categories, storing maximum probability scores to allow for multi-dimensional metadata filtering.

### **04. FastAPI Backend**
* **Production Routing:** Engineered a high-throughput REST API to serve trained ML inference pipelines and vector retrieval.
* **Structured Outputs:** Utilized strict Pydantic schemas (`QueryRequest`, `QueryResponse`) for bulletproof input validation and predictable JSON responses.

### **05. LLM-as-a-Judge Evaluation Framework**
* **Automated Benchmarks:** Built a rigorous evaluation script (`eval_harness.py`) that programmatically hits the local API with diverse test queries.
* **AI Observability:** Utilizes Claude 4.5 Haiku to analyze retrieved contexts, score semantic relevance (0.0 - 1.0), and successfully verify a 0% hallucination rate across the database.

---

## 🔧 Technical Challenges & Solutions

Building **Biblios** involved solving several infrastructure and deployment obstacles:

* **Vector Dilution & Chunking Constraints:** Re-architected LangChain's `CharacterTextSplitter` logic. Shifting from 10,000-character blob chunking to isolated document instantiation drastically improved the LLM judge's relevance scores.
* **Database Concurrency & Locking:** Engineered clean teardown and instantiation logic for ChromaDB to prevent persistent SQLite file-lock (code: 14) errors during rapid iterative development and testing.
* **Erratic LLM Formatting:** Implemented resilient string parsing in the evaluation harness to strip unexpected markdown blockquotes and enforce strict JSON loading from the Anthropic API.
* **Data Parsing Issues:** Fixed a `ValueError` during ISBN retrieval where CSV export quotes were interfering with integer conversion.

---

## ⚙️ Installation & Setup

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/amriikk/Biblios.git](https://github.com/amriikk/Biblios.git)
    cd Biblios
    ```

2.  **Install Production Dependencies:**
    ```bash
    pip install fastapi uvicorn pydantic pandas langchain-huggingface langchain-chroma sentence-transformers python-dotenv anthropic
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory and add your evaluation API key:
    ```env
    ANTHROPIC_API_KEY="your-api-key-here"
    ```

4.  **Launch the API Server:**
    ```bash
    python -m uvicorn src.main:app --reload
    ```
    *Navigate to `http://127.0.0.1:8000/docs` to interact with the Swagger UI.*

5.  **Run the Evaluation Harness:**
    Open a second terminal and execute the automated benchmarks:
    ```bash
    python src/eval_harness.py
    ```

---

> **Note:** This project demonstrates practical Applied AI implementation, focusing on turning probabilistic models into deterministic, fault-tolerant tools ready for production environments.