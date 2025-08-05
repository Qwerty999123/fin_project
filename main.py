import os
from fastapi import FastAPI, Request, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from google import genai
from semantic_aware import load_document
import hashlib
import httpx
from datetime import datetime
import re
from pinecone import Pinecone
from pinecone_embeddings import PineconeVectorStore


# Configuration
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
PINECONE_INDEX = 'policy-documents'
CACHE_DIR = "./document_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
pinecone = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Initialize Gemini
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def document_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

async def fetch_with_cache(url: str) -> str:
    """Download with caching"""
    cache_key = document_cache_key(url)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pdf")
    
    if os.path.exists(cache_path):
        return cache_path
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(response.content)
    
    return cache_path

def build_gemini_prompt(question: str, clauses: List[dict]) -> str:
    """Strictly formatted prompt for Gemini"""
    context = "\n\n".join(
        f"CLAUSE {c.get('header', '')} (Page {c.get('page', 'N/A')}):\n{c['text']}"
        for c in clauses
    )
    
    return f"""You are a strict, accurate assistant that answers insurance or policy-related questions using only provided clauses.

        A user has asked the following question:
        "{question}"

        You must answer only based on the given text below, without guessing or skipping any information.
        If an answer is partially stated or implied, respond accordingly with brief clarification.
        If the information is not present at all, reply exactly: "Not mentioned in the provided clauses."

        Clauses:
        {context}

        Respond with 1 to 3 sentences max.
        Do not add explanations, formatting, bullet points, summaries, or any output other than the answer sentence.

"""

# def extract_first_sentence(text: str) -> str:
#     """Ensure single-sentence output"""
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     return sentences[0] if sentences else text

@app.post("/query", response_model=QueryResponse)
async def answer_questions(request: Request, body: QueryRequest):
    # Authentication
    if request.headers.get("Authorization") != f"Bearer {os.environ['API_KEY']}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    try:
        # 1. Process document
        local_path = await fetch_with_cache(body.documents)
        doc = load_document(local_path)
        
        # 2. Initialize engine
        vector_store = PineconeVectorStore(index_name=PINECONE_INDEX, pinecone=pinecone)
        vector_store.overwrite_vectors(doc["chunks"], 'doc_a.pdf', pinecone)
        
        # 3. Process questions
        answers = []
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        for question in body.questions:
            # Retrieve relevant clauses
            clauses = vector_store.retrieve_chunks(question, pinecone, top_k=5)
        
            # print("\n\n")
            # print(clauses)

            # Generate answer with Gemini
            prompt = build_gemini_prompt(question, clauses)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            # Strict formatting
            # answer = extract_first_sentence(response.text)
            # print(response.text)
            answers.append(response.text)
        
        return {"answers": answers}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )