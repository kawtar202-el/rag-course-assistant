import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import COURSE_MATERIALS_DIR

def load_documents(data_dir=COURSE_MATERIALS_DIR):
    """
    Load documents from the specified directory
    Supports PDF, TXT, and DOCX files
    """
    documents = []
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks for better processing
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks