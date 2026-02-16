"""
Configuration management for EnteAI
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
VECTOR_INDEX_PATH = os.path.join(STORAGE_DIR, "vector.index")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(STORAGE_DIR, "metadata.pkl")

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Text processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Search configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# LLM configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))

# File upload limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = [".txt", ".pdf"]

# UI configuration
ENABLE_TYPING_EFFECT = os.getenv("ENABLE_TYPING_EFFECT", "true").lower() == "true"
TYPING_SPEED = float(os.getenv("TYPING_SPEED", "0.01"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "enteai.log")
