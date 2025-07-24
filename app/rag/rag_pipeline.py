from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings
from app.rag.document_loader import load_documents
from app.rag.llm_chain import create_rag_chain
from app.rag.vector_store import create_vector_store

class RAGSystem:
    def __init__(self):
        """Initializes the RAG system by loading documents and setting up the chain."""
        self.chat_history = []
        self.retrieval_chain = None
        self._initialize_rag()

    def _initialize_rag(self):
        """
        Loads documents from the data path, creates a vector store,
        and initializes the RAG retrieval chain. This runs once at startup.
        """
        print("Initializing RAG System...")
        
        # 1. Load documents from the predefined 'data' folder.
        print(f"Loading documents from: {settings.DATA_PATH}")
        docs = load_documents(settings.DATA_PATH)
        if not docs:
            print("Warning: No documents found in the 'data' folder. The system will only work if documents are uploaded via the API.")
            return

        # 2. Create a vector store from the loaded documents.
        print("Creating vector store...")
        vector_store = create_vector_store(docs)
        if vector_store is None:
            print("Error: Vector store could not be created. Halting RAG chain initialization.")
            return

        # 3. Create the RAG chain and store it in the instance.
        print("Creating RAG chain...")
        self.retrieval_chain = create_rag_chain(vector_store)
        print("RAG System Initialized Successfully.")

# Initialize the RAG system globally.
# This instance holds the RAG chain for any documents present in the 'data' folder at startup.
rag_system = RAGSystem()
