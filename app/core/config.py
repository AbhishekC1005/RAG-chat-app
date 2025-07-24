import os
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root.
load_dotenv()

class Settings:
    """
    Application settings, loaded from environment variables.
    """
    # Groq API Key for LLM inference.
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    # Path to the data directory where documents are pre-loaded.
    DATA_PATH = os.getenv("DATA_PATH", "data")
    
    # Embedding model from HuggingFace.
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    
    # LLM Model for Groq.
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
    
    # Text splitter settings for document chunking.
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

settings = Settings()
