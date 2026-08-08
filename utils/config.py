"""
config.py
---------
Central configuration for the RAG application.

All tunable parameters live here and can be overridden via environment
variables (loaded from a .env file). Keeping configuration in one place
makes the rest of the codebase easier to read and modify.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file into the process environment, if present.
load_dotenv()
RETRIEVAL_SCORE_THRESHOLD = 0.5
# --------------------------------------------------------------------------
# API / Model settings (Google Gemini)
# --------------------------------------------------------------------------
# langchain-google-genai also accepts GEMINI_API_KEY automatically, but we
# read GOOGLE_API_KEY explicitly here so we can give a clear error message
# if it's missing.
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Chat model used to generate answers.
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Embedding model used to vectorize document chunks and queries.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Temperature of 0 keeps answers deterministic and grounded (less "creative").
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# --------------------------------------------------------------------------
# Chunking settings (bonus: configurable via env vars)
# --------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# --------------------------------------------------------------------------
# Retrieval settings
# --------------------------------------------------------------------------
# Number of most-similar chunks to retrieve for each question.
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "3"))

# Whether to print the similarity score of each retrieved chunk (bonus).
SHOW_SIMILARITY_SCORES: bool = os.getenv("SHOW_SIMILARITY_SCORES", "true").lower() == "true"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR: str = os.path.join(BASE_DIR, "documents")
DB_DIR: str = os.path.join(BASE_DIR, "db")

# Manifest file used to detect whether documents changed since the last
# ingestion run (so we don't needlessly regenerate embeddings).
MANIFEST_PATH: str = os.path.join(DB_DIR, "manifest.json")

# Name of the Chroma collection.
COLLECTION_NAME: str = "rag_documents"

# --------------------------------------------------------------------------
# Supported file types (bonus: TXT support in addition to PDF)
# --------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".pdf", ".txt")
