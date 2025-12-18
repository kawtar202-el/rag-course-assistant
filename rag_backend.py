import requests
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from config import HUGGINGFACEHUB_API_TOKEN, MODEL_URL
from pydantic import Field
from huggingface_hub import InferenceClient

class HuggingFaceInferenceLLM(LLM):
    """
    Custom LLM class that uses Hugging Face Inference API directly
    """
    # Use MODEL_URL from env if present, otherwise default to Llama model
    # Using Llama which is available via Hugging Face router
    model_name: str = Field(default="meta-llama/Llama-3.1-8B-Instruct:novita")  # Default model name
    api_token: str = Field(default="")

    def __init__(self, **kwargs):
        api_token = HUGGINGFACEHUB_API_TOKEN
        # Allow overriding the model via constructor kwargs or env var
        provided_model = kwargs.pop('model_name', None) or MODEL_URL or "meta-llama/Llama-3.1-8B-Instruct:novita"
        super().__init__(model_name=provided_model, api_token=api_token, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface_inference"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Format the prompt for the Llama model using the router API chat completion format
        # Based on the example provided for Hugging Face router
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model_name,  # Include the model in the payload as shown in the example
            "max_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.9
        }

        # Use the Hugging Face Router API endpoint - the example shows this path
        api_url = "https://router.huggingface.co/v1/chat/completions"

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60  # Increase timeout for model loading
            )

            # Handle the case where model needs to be loaded first (503 error) or other issues
            if response.status_code == 503:
                print(f"[HF DEBUG] Model {self.model_name} is currently loading, please wait...")
                print(f"[HF DEBUG] Response: {response.text}")
                # Retry after a short delay if model is loading
                time.sleep(10)  # Wait 10 seconds

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

            if response.status_code == 200:
                result = response.json()

                # Handle chat completion response format
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print(f'[HF DEBUG] Unexpected response format: {result}')
                    return str(result)
            else:
                print(f"[HF DEBUG] API request failed with status {response.status_code}")
                print(f"[HF DEBUG] Response: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[HF DEBUG] Request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")

def get_llm():
    """
    Initialize the Hugging Face LLM with custom implementation
    Using a model through Hugging Face Inference API
    """
    llm = HuggingFaceInferenceLLM()
    return llm

def get_embeddings():
    """
    Initialize Hugging Face sentence transformer embeddings
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Lightweight embedding model
    )
    return embeddings

def create_prompt_template():
    """
    Create a prompt template for question answering
    """
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt