import os
import requests
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Initialize Anthropic Client for the LLM-as-a-judge
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

API_URL = "http://127.0.0.1:8000/api/v1/recommend"

# A batch of diverse test queries to benchmark the retrieval system
EVAL_QUERIES = [
    "A story about a troubled family set across many generations",
    "A book to teach children about nature",
    "A suspenseful novel about a detective solving a cold case",
    "A comprehensive history of the Roman Empire",
    "A humorous science fiction adventure in space"
]

class EvalScore(BaseModel):
    query: str
    relevance_score: float # 0.0 to 1.0
    hallucination_detected: bool
    judge_reasoning: str

def get_recommendations(query: str, top_k: int = 3) -> List[Dict]:
    """Hits your local FastAPI server to retrieve semantic recommendations."""
    payload = {"query": query, "top_k": top_k}
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()["results"]

def evaluate_retrieval(query: str, results: List[Dict]) -> EvalScore:
    """Uses Claude to judge the relevance of the retrieved book descriptions."""
    
    # Format the retrieved descriptions into a prompt for the LLM judge
    context = "\n\n".join([f"Book {i+1}:\n{res['description']}" for i, res in enumerate(results)])
    
    prompt = f"""
    You are an expert retrieval-evaluation system. 
    A user searched for: "{query}"
    
    The semantic search engine returned the following book descriptions:
    {context}
    
    Evaluate the results based on two criteria:
    1. Relevance Score (0.0 to 1.0): How accurately do these books match the user's query intent? 
    2. Hallucination: Did the system return any books that are completely irrelevant or contradictory to the query?
    
    Output your evaluation strictly in the following JSON format:
    {{
        "relevance_score": 0.0,
        "hallucination_detected": false,
        "judge_reasoning": "brief explanation"
    }}
    """
    
    response = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        temperature=0.0,
        system="You output strict JSON without markdown formatting.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse the JSON response into our Pydantic model
    import json
    eval_data = json.loads(response.content[0].text)
    
    return EvalScore(
        query=query,
        relevance_score=eval_data["relevance_score"],
        hallucination_detected=eval_data["hallucination_detected"],
        judge_reasoning=eval_data["judge_reasoning"]
    )

def run_evaluation_suite():
    print("Starting LLM-as-a-judge evaluation suite...\n")
    all_scores = []
    
    for query in EVAL_QUERIES:
        print(f"Testing Query: '{query}'")
        try:
            # 1. Hit the local API
            results = get_recommendations(query)
            
            # 2. Score with Claude
            score = evaluate_retrieval(query, results)
            all_scores.append(score.model_dump())
            
            print(f"  Relevance: {score.relevance_score}")
            print(f"  Hallucination: {score.hallucination_detected}")
            print(f"  Reasoning: {score.judge_reasoning}\n")
            
        except Exception as e:
            print(f"  Error evaluating query: {e}\n")
            
    # Compile final metrics
    df = pd.DataFrame(all_scores)
    avg_relevance = df['relevance_score'].mean()
    total_hallucinations = df['hallucination_detected'].sum()
    
    print("--- Final Benchmark Metrics ---")
    print(f"Average Relevance Score: {avg_relevance:.2f}/1.0")
    print(f"Total Hallucinations Detected: {total_hallucinations}/{len(EVAL_QUERIES)}")

if __name__ == "__main__":
    run_evaluation_suite()