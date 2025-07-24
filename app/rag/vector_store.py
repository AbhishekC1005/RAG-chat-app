import os
import shutil
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

# Define the path to the data directory, which is one level up from the 'app' directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def get_vector_store_path(file_hash: str) -> str:
    """Helper function to get the full path for a vector store based on a file hash."""
    return os.path.join(DATA_DIR, f'vectorstore_{file_hash}')

def save_vector_store(vector_store, file_hash: str):
    """Saves the vector store to a local path."""
    path = get_vector_store_path(file_hash)
    vector_store.save_local(path)
    print(f"Saved new vector store for hash: {file_hash}")

def load_vector_store(file_hash: str, embeddings):
    """Loads the vector store from a local path if it exists."""
    path = get_vector_store_path(file_hash)
    if os.path.exists(path):
        print(f"Loaded existing vector store for hash: {file_hash}")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None

def delete_vector_store(file_hash: str):
    """Deletes a vector store directory if it exists."""
    path = get_vector_store_path(file_hash)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted old vector store for hash: {file_hash}")

def create_vector_store(documents: List[Document], file_hash: Optional[str] = None):
    """
    Creates a new FAISS vector store or loads an existing one from a list of documents.
    Uses a file_hash for caching the vector store.
    """
    if not documents:
        print("No documents provided to create_vector_store. Returning None.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # If a file_hash is provided, try to load the cached vector store first
    if file_hash:
        vector_store = load_vector_store(file_hash, embeddings)
        if vector_store:
            return vector_store

    # If no cached version exists, create a new one
    print("Creating new vector store...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Save the new vector store if a file_hash was provided
    if file_hash:
        save_vector_store(vector_store, file_hash)

    return vector_store
