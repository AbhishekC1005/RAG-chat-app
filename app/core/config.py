import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Settings:
    """
    Application settings.
    """
    # Groq API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    # Path to the data directory
    DATA_PATH = os.getenv("DATA_PATH", "data")
    
    # Embedding model (Groq currently does not provide embeddings, so use OpenAI or another provider if needed)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # LLM Model (Groq supported model, e.g., 'llama3-8b-8192')
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
    
    # Text splitter settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

settings = Settings()
