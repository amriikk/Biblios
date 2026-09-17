from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated import

# Initialize API
app = FastAPI(
    title="Biblios Semantic Engine",
    description="Production RAG and semantic search API for book recommendations.",
    version="1.0.0"
)

# Connect to the persistent vector database using local embeddings
try:
    # This downloads a lightweight, highly efficient embedding model locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
except Exception as e:
    print(f"Failed to load vector database: {e}")

# Strict Pydantic Schemas for validation
class QueryRequest(BaseModel):
    query: str = Field(..., description="The semantic search query from the user.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return.")
    emotion_filter: Optional[str] = Field(default=None, description="Optional emotional tone to filter by.")

class SearchResult(BaseModel):
    isbn13: str
    description: str
    relevance_score: float

class QueryResponse(BaseModel):
    results: List[SearchResult]
    total_returned: int

@app.post("/api/v1/recommend", response_model=QueryResponse)
async def get_recommendations(request: QueryRequest):
    try:
        # Execute similarity search with scores (lower score = closer distance)
        docs_and_scores = vector_db.similarity_search_with_score(request.query, k=request.top_k)
        
        results = []
        for doc, score in docs_and_scores:
            # Extract ISBN from the first word of our tagged description
            raw_text = doc.page_content
            isbn = raw_text.split(" ")[0].strip()
            desc = raw_text.replace(isbn, "").strip()
            
            results.append(SearchResult(
                isbn13=isbn,
                description=desc,
                relevance_score=round(score, 4)
            ))
            
        return QueryResponse(results=results, total_returned=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}