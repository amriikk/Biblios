import gradio as gr
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the cleaned dataset and local vector DB
books = pd.read_csv("books_cleaned.csv")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def retrieve_recommendations(query: str, top_k: int = 5):
    """Retrieves books and formats them into a visual gallery display."""
    if not query.strip():
        return []
    
    docs_and_scores = vector_db.similarity_search_with_score(query, k=top_k)
    results = []
    
    for doc, score in docs_and_scores:
        raw_text = doc.page_content
        isbn = raw_text.split(" ")[0].strip().replace('"', '')
        
        # Match back to dataframe for rich metadata
        match = books[books["isbn13"] == int(isbn)]
        if not match.empty:
            row = match.iloc[0]
            title = row.get("title", "Unknown Title")
            author = row.get("authors", "Unknown Author")
            
            # Ensure thumbnail is a valid string; if float/NaN, set to None
            thumbnail = row.get("thumbnail", None)
            if pd.isna(thumbnail) or not isinstance(thumbnail, str):
                thumbnail = None
                
            desc = row.get("description", "")[:250] + "..."
            
            caption = f"**{title}** by {author}\n\n{desc}\n\n*(Distance: {score:.3f})*"
            results.append((thumbnail, caption))
            
    return results

# Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Biblios: Semantic Book Discovery Engine")
    gr.Markdown("Search for books using natural language. The system maps your intent to high-dimensional vector embeddings for context-aware recommendations.")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Search Query", 
            placeholder="e.g., A story about a troubled family set across many generations",
            scale=4
        )
        submit_btn = gr.Button("Find Recommendations", variant="primary", scale=1)
        
    gallery = gr.Gallery(
        label="Recommended Books", 
        columns=2, 
        rows=2, 
        object_fit="contain"
    )
    
    submit_btn.click(fn=retrieve_recommendations, inputs=[query_input], outputs=[gallery])
    query_input.submit(fn=retrieve_recommendations, inputs=[query_input], outputs=[gallery])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Glass())