from langchain_community.vectorstores import FAISS
from document_processor import load_documents, chunk_documents
from rag_backend import get_embeddings
import os
import pickle

class VectorStoreManager:
    def __init__(self, data_dir="course_materials", vector_store_path="vector_store"):
        self.data_dir = data_dir
        self.vector_store_path = vector_store_path
        self.embeddings = get_embeddings()
        
    def create_vector_store(self):
        """
        Create a vector store from course materials
        """
        # Load and chunk documents
        raw_documents = load_documents(self.data_dir)
        documents = chunk_documents(raw_documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save vector store locally
        vector_store.save_local(self.vector_store_path)
        
        return vector_store
    
    def load_vector_store(self):
        """
        Load existing vector store or create if it doesn't exist
        """
        if os.path.exists(self.vector_store_path):
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            vector_store = self.create_vector_store()
        
        return vector_store