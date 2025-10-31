import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(pdf_path: str) -> List[Document]:
    """Loads the PDF document."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    return loader.load()