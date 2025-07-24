RAG Decision API
This project implements a Retrieval-Augmented Generation (RAG) system for intelligent decision-making based on unstructured documents like contracts, policies, and emails.

Features
FastAPI Backend: A robust and fast API for serving the model.

LangChain & Groq: Leverages LangChain for orchestration and the fast Groq LPU for inference.

Dynamic Document Handling: Process documents uploaded via API or pre-loaded in a data folder.

Semantic Search: Uses FAISS vector stores for efficient, meaning-based retrieval of relevant clauses.

Structured Output: Returns detailed, structured JSON responses with decisions, justifications, and clause mappings.

Project Structure
.
├── app/
│   ├── core/
│   │   └── config.py
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── llm_chain.py
│   │   ├── rag_pipeline.py
│   │   └── vector_store.py
│   └── main.py
├── data/
│   └── (Add your policy documents here)
├── .env
├── .gitignore
├── main.py         <-- Run this file
└── requirements.txt

Setup Instructions
1. Clone the Repository

git clone <your-repository-url>
cd <repository-name>

2. Create and Activate a Virtual Environment

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Environment Variables
Create a file named .env in the project root and add your Groq API key:

GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"

5. (Optional) Pre-load Documents
Place any policy documents (PDF, DOCX, etc.) you want to be available at startup into the data folder.

6. Run the Application

python main.py

The API will be running at http://127.0.0.1:8000.

API Usage
Navigate to http://127.0.0.1:8000/docs in your browser to access the interactive FastAPI documentation.

Example curl Request
You can test the /decision endpoint from your terminal:

With a pre-loaded document:

curl -X POST "http://localhost:8000/decision" \
-H "Content-Type: multipart/form-data" \
-F "query=Is a 46-year-old male covered for knee surgery if the policy is 3 months old?"

With an uploaded document:

curl -X POST "http://localhost:8000/decision" \
-H "Content-Type: multipart/form-data" \
-F "query=Is cosmetic surgery covered?" \
-F "file=@/path/to/your/policy.pdf"
