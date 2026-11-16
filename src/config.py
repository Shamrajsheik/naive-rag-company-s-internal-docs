import os
from pathlib import Path
from dotenv import load_dotenv

from src.generator import (
    DEFAULT_GROQ_MODEL,
    FALLBACK_GROQ_MODEL,
    GROQ_BASE_URL,
    DEFAULT_TEMPERATURE,
    STRICT_SYSTEM_PROMPT,
)
from src.loader import BASE_DIR, COMPANY_DOCS_DIR, resolve_docs_dir

# Load optional local .env if present (not required)
load_dotenv()

# In-code directory paths
DOCS_DIR = resolve_docs_dir()
DEFAULT_CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_PERSIST_DIR = DEFAULT_CHROMA_DIR
COLLECTION_NAME = "company_docs"

# Model and API settings
GROQ_MODEL = DEFAULT_GROQ_MODEL
FALLBACK_MODEL = FALLBACK_GROQ_MODEL
DEFAULT_TOP_K = 3
