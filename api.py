"""
FastAPI backend for RAG Course Assistant web interface
Connects the HTML chat interface to the RAG system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_backend import get_llm, create_prompt_template
from vector_store import VectorStoreManager
from langchain.chains import RetrievalQA

app = FastAPI(title="RAG Course Assistant API", description="API for RAG Course Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the QA chain
qa_chain = None

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

def initialize_rag_system():
    """Initialize the RAG system components"""
    global qa_chain
    
    if qa_chain is None:
        print("Initializing RAG system...")
        
        # Initialize vector store
        vector_store_manager = VectorStoreManager()
        vector_store = vector_store_manager.load_vector_store()
        
        # Initialize LLM
        llm = get_llm()
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False,  # Don't return source documents for clean UI
            chain_type_kwargs={"prompt": create_prompt_template()}
        )
        print("RAG system initialized successfully!")

@app.on_event("startup")
async def startup_event():
    initialize_rag_system()

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint to ask a question and get an answer"""
    try:
        # Process the question with RAG system
        response = qa_chain({"query": request.question})
        
        return AnswerResponse(
            question=request.question,
            answer=response["result"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG Course Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)