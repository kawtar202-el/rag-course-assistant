import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Changed to standard env var name

MODEL_URL = os.getenv("MODEL_URL") or "meta-llama/Llama-3.1-8B-Instruct:novita"

# Directory for course materials
COURSE_MATERIALS_DIR = "course_materials"
OUTPUT_DIR = "output"