# Biblios 📚
### An Intelligent Semantic Discovery Engine

**Biblios** is a high-performance book recommendation system that transcends traditional keyword-based search. By leveraging **Large Language Models (LLMs)** and **Vector Databases**, it understands the deeper context, themes, and emotional resonance of a narrative to provide truly relevant discoveries.

---

## 🛠️ The Tech Stack

* **Language:** Python 3.13
* **AI/LLM Frameworks:** LangChain, Hugging Face Transformers
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local Execution)
* **Vector Database:** ChromaDB (via `langchain-chroma`)
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
* **UI Framework:** Gradio
* **Secrets Management:** `python-dotenv`

---

## 🚀 Project Architecture

The project is structured into five distinct engineering phases:

### **01. Prepare Text Data**
* **Exploratory Data Analysis (EDA):** Performed correlation analysis using Spearman heatmaps to ensure data missingness was not biased.
* **Data Cleaning:** Engineered features like `age_of_book` and `title_and_subtitle` while filtering for semantic density (25-word minimum description threshold).
* **Tagging:** Prepended unique ISBN identifiers to descriptions to allow for precise metadata retrieval after vector matching.

### **02. Vector Search (The "Brain")**
* **Semantic Embeddings:** Swapped OpenAI for a locally-hosted Hugging Face model (`all-MiniLM-L6-v2`) to generate high-dimensional vectors for 5,000+ books.
* **Indexing:** Utilized **ChromaDB** for efficient K-Nearest Neighbor (KNN) retrieval.
* **Query Logic:** Implemented a similarity search that translates natural language (e.g., *"a story about redemption in the Arctic"*) into mathematical coordinates to find thematic neighbors.

### **03. Text Classification**
* **Zero-Shot Learning:** Implemented `facebook/bart-large-mnli` to classify books into categories (Fiction vs. Non-Fiction) without the need for pre-labeled training data.
* **Data Augmentation:** Used model inference to fill gaps in the original dataset's labeling.

### **04. Sentiment Analysis**
* **Emotion Mapping:** Leveraged a fine-tuned **RoBERTa** model to detect sentence-level sentiment across seven categories (Joy, Fear, Sadness, etc.).
* **Scoring:** Stored maximum probability scores to allow users to sort recommendations by their desired "vibe."

### **05. Gradio Dashboard**
* **Interactive UI:** A "Glass" themed dashboard that provides a polished gallery view of book covers and metadata.
* **Dynamic Controls:** Users can combine semantic queries with category filters and emotional tone sorting.

---

## 🔧 Technical Challenges & Solutions

Building **Biblios** involved solving several real-world engineering obstacles:

* **The Persistence Problem:** Encountered duplicate entries in ChromaDB during re-indexing. Solved by implementing a manual collection wipe/reset logic before indexing runs.
* **Data Parsing Issues:** Fixed a `ValueError` during ISBN retrieval where CSV export quotes were interfering with integer conversion.
* **Version Mismatches:** Resolved `langchain` and `pandas` deprecation warnings (specifically the `sep` argument in `to_csv` and `chunk_size` constraints in `CharacterTextSplitter`) by upgrading to modern syntax.

---

## ⚙️ Installation & Setup

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/yourusername/biblios.git](https://github.com/yourusername/biblios.git)
    cd biblios
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas seaborn matplotlib langchain-huggingface langchain-chroma sentence-transformers python-dotenv gradio
    ```

3.  **Environment Variables:**
    Create a `.env` file (see `.env.example` for the template). **Note:** This project is configured to run Hugging Face embeddings locally, reducing API dependency.

---

## 📊 Visuals
*(Note to Self: Insert your Spearman Correlation Heatmap here to show off the EDA!)*

---

> **Note:** This project was developed as a deep dive into Semantic Search and Agentic AI workflows.