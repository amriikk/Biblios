# Biblios: Semantic Book Discovery Engine

**Biblios** is an intelligent book recommendation system that moves beyond simple keyword matching. By leveraging **Large Language Models (LLMs)** and **Vector Databases**, it understands the "soul" of a book—its themes, emotional resonance, and narrative style—to provide deeply relevant suggestions.

---

## 🛠️ The Tech Stack

* **Language:** Python
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
* **AI/LLM Frameworks:** LangChain, Transformers (Hugging Face), OpenAI
* **Vector Store:** ChromaDB (via `langchain-chroma`)
* **Interface:** Gradio
* **Environment Management:** python-dotenv

---

## 🚀 Project Architecture

The project is divided into five core phases:

### **01. Prepare Text Data**
* **Cleaning:** Handled missing values and performed feature engineering (e.g., `age_of_book`).
* **Filtering:** Implemented a 25-word minimum threshold for descriptions to ensure sufficient semantic density for the LLM.
* **Feature Engineering:** Merged titles and subtitles into a unified `title_and_subtitle` column and created unique "Tag Descriptions" by prepending ISBNs for efficient database indexing.

### **02. Vector Search (Semantic Engine)**
* **Embeddings:** Converted raw text into high-dimensional vectors using OpenAI's embedding models.
* **Vector Store:** Indexed the cleaned dataset in **ChromaDB** to enable efficient similarity searching.
* **Retrieval:** Developed a semantic search function that identifies the "mathematical neighbors" of a user's natural language query.

### **03. Text Classification**
* **Zero-Shot Learning:** Utilized the `facebook/bart-large-mnli` model to categorize books into Fiction vs. Non-Fiction without the need for explicit training data.
* **Data Augmentation:** Used high-confidence AI predictions to fill gaps in the original dataset's categorization, ensuring a complete filtering experience for the user.

### **04. Sentiment Analysis**
* **Emotion Detection:** Leveraged a fine-tuned **RoBERTa** model to extract emotional tones such as Joy, Sadness, Surprise, Fear, and Anger.
* **Nuanced Analysis:** Analyzed sentiment at the sentence level to capture the complex emotional "vibe" of a book, storing the maximum probability scores for each emotion.

### **05. Gradio Dashboard**
* Created a sleek, user-friendly UI using the **Gradio** "Glass" theme.
* **Features:**
    * **Natural Language Input:** Users can describe the *kind* of story they want.
    * **Dynamic Filtering:** Filter results by AI-defined categories.
    * **Emotional Sorting:** Sort recommendations by specific "vibe" (e.g., Happy, Suspenseful, Sad).

---

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/biblios.git](https://github.com/yourusername/biblios.git)
    cd biblios
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas seaborn matplotlib langchain transformers python-dotenv gradio openai langchain-chroma langchain-openai
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_actual_key_here
    ```

---

## 📊 Visuals & Insights
*(Insert your correlation heatmap or Gradio interface screenshots here to showcase the project!)*

---

> **Note:** This project was developed as a deep dive into Semantic Search methodologies and the practical application of LLMs in content discovery.