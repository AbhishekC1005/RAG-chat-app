from app.core.config import settings
from app.rag.document_loader import load_documents
from app.rag.vector_store import create_vector_store
from app.rag.llm_chain import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from langchain_core.documents import Document

class RAGSystem:
    def __init__(self):
        self.chat_history = []
        self._initialize_rag()

    def _initialize_rag(self):
        """Initializes the RAG system."""
        print("Initializing RAG System...")
        print("Loading documents...")
        docs = load_documents(settings.DATA_PATH)
        if not docs:
            print("No documents loaded. The chatbot will rely solely on its pre-trained knowledge.")
            self.retrieval_chain = None
            return
        
        print("Creating vector store...")
        vector_store = create_vector_store(docs)
        if vector_store is None:
            print("Vector store could not be created. Skipping RAG chain initialization.")
            self.retrieval_chain = None
            return
        
        print("Creating RAG chain...")
        self.retrieval_chain = create_rag_chain(vector_store)
        print("RAG System Initialized.")


    def get_response(self, user_input: str):
        """Gets a response from the RAG system for a given user input."""
        if not self.retrieval_chain:
            return "RAG system is not initialized due to missing documents. Please add documents to the data folder."
        response = self.retrieval_chain.invoke({
            "chat_history": self.chat_history,
            "input": user_input
        })
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

# Initialize the RAG system globally.
# This will be created once when the application starts.
rag_system = RAGSystem()
