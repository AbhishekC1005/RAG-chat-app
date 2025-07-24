import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from typing import List
from langchain_core.documents import Document

def load_documents(data_path: str) -> List[Document]:
    """
    Loads documents from the specified data path.
    Supports .pdf, .docx, .eml, and .msg files.
    """
    docs = []
    if not os.path.isdir(data_path):
        print(f"Error: Data path '{data_path}' not found or is not a directory.")
        return docs

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs.extend(loader.load())
            elif file.endswith(".eml") or file.endswith(".msg"):
                loader = UnstructuredEmailLoader(file_path)
                docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
    return docs
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from typing import List
from langchain_core.documents import Document

def load_documents(data_path: str) -> List[Document]:
    """
    Loads documents from the specified data path.
    Supports .pdf, .docx, .eml, and .msg files.
    """
    docs = []
    if not os.path.isdir(data_path):
        print(f"Warning: Data path '{data_path}' not found or is not a directory.")
        return docs

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        try:
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif file.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs.extend(loader.load())
            elif file.lower().endswith((".eml", ".msg")):
                loader = UnstructuredEmailLoader(file_path)
                docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
    return docs
