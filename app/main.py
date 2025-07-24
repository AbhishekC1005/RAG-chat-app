import os
import hashlib
from typing import Optional, List

# --- FastAPI Imports ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Pydantic and LangChain Imports ---
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq

# --- Application-Specific Imports ---
from app.core.config import settings
from app.rag.rag_pipeline import rag_system
from app.rag.document_loader import load_documents
from app.rag.vector_store import create_vector_store, delete_vector_store
from app.rag.llm_chain import create_rag_chain

# ==============================================================================
# 1. PYDANTIC MODELS FOR STRUCTURED DATA
# ==============================================================================

class ClauseMapping(BaseModel):
    """Maps a policy clause to its relevance for the decision."""
    clause: Optional[str] = Field(None, description="The exact policy clause referenced.")
    reason: Optional[str] = Field(None, description="The reason this clause is relevant to the decision.")

class DecisionResponse(BaseModel):
    """The final, structured API response model."""
    answer: Optional[str] = Field(None, description="A concise, one-line answer to the user's query.")
    decision: Optional[str] = None
    amount: Optional[float] = None
    justification: Optional[str] = None
    clause_mapping: Optional[List[ClauseMapping]] = None

class StructuredQuery(BaseModel):
    """Structured representation of the user's initial query."""
    age: Optional[int] = Field(None, description="The age of the person in the query.")
    procedure: Optional[str] = Field(None, description="The medical procedure mentioned.")
    location: Optional[str] = Field(None, description="The location where the procedure occurred.")
    policy_duration: Optional[str] = Field(None, description="The duration the policy has been active, e.g., '3 months'.")

class DecisionOutput(BaseModel):
    """The schema for the LLM's decision-making output."""
    decision: Optional[str] = Field(None, description="The final decision, either 'Approved' or 'Rejected'.")
    amount: Optional[float] = Field(None, description="The approved payout amount as a number. Null if rejected.")
    justification: Optional[str] = Field(None, description="A high-level summary explaining the final decision.")
    clause_mapping: Optional[List[ClauseMapping]] = Field(None, description="A list of specific clauses used for the decision.")

    @field_validator("amount", mode='before')
    @classmethod
    def validate_amount(cls, v):
        """Allows string-to-float conversion and handles invalid values."""
        if v is None or isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None
        return v

# ==============================================================================
# 2. FASTAPI APPLICATION SETUP
# ==============================================================================

app = FastAPI(
    title="RAG Decision API",
    description="API for insurance/legal/contract decisioning with explainability using FastAPI, LangChain, and Groq.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 3. API ENDPOINT LOGIC
# ==============================================================================

@app.post("/decision", response_model=DecisionResponse)
async def decision_api(query: str = Form(...), file: Optional[UploadFile] = File(None)):
    """
    Accepts a user query and an optional document, processes it, and returns a structured decision.
    """
    # --- Step 1: Determine which RAG chain to use ---
    rag_chain = None
    temp_dir = "temp_docs"

    if file:
        # If a file is uploaded, create a temporary RAG chain just for this request.
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        try:
            file_bytes = await file.read()
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            docs = load_documents(temp_dir)
            if not docs:
                raise HTTPException(status_code=400, detail="Could not read the uploaded document.")
            
            vector_store = create_vector_store(docs)
            if not vector_store:
                raise HTTPException(status_code=500, detail="Could not create vector store for the uploaded document.")
            
            rag_chain = create_rag_chain(vector_store)
        finally:
            # Clean up the temporary file and directory
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
    else:
        # If no file is uploaded, use the globally initialized RAG chain.
        rag_chain = rag_system.retrieval_chain

    if not rag_chain:
        raise HTTPException(status_code=400, detail="No documents available for decisioning. Please upload a document or start the service with documents in the 'data' folder.")

    # --- Step 2: Process the query using the selected RAG chain ---
    llm = ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY)

    try:
        # First, get a direct answer and the context used.
        rag_result = await rag_chain.ainvoke({"chat_history": [], "input": query})
        initial_answer = rag_result.get("answer", "").strip()
        retrieved_clauses = rag_result.get("context", "")

        # Handle cases where the document doesn't contain the answer.
        if "don't know" in initial_answer.lower() or "unable to answer" in initial_answer.lower():
            return JSONResponse(content={"answer": "I don't know."})

        # Parse the natural language query into a structured format.
        structured_query = await llm.with_structured_output(StructuredQuery).ainvoke(query)

        # --- Step 3: Generate the final structured decision ---
        decision_prompt = (
            "You are an expert insurance claim analyst. Based on the user's query and the provided policy clauses, "
            "make a final decision. Your response must conform to the required JSON structure."
            f"\n\n## User Query Details:\n{structured_query.model_dump_json(indent=2)}"
            f"\n\n## Relevant Policy Clauses:\n{retrieved_clauses}"
            "\n\n## Instructions:\nReturn a JSON object for your decision. The 'amount' field MUST be a number (e.g., 5000.0 or 0)."
        )

        decision_result = await llm.with_structured_output(DecisionOutput).ainvoke(decision_prompt)
        final_response_data = decision_result.model_dump()
        final_response_data['answer'] = initial_answer # Add the concise answer
        
        return DecisionResponse(**final_response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM processing: {e}")

@app.get("/")
def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the RAG Decision API. Use the /docs endpoint for documentation."}
