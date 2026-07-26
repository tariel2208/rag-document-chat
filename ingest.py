"""
ingest.py
---------
Document ingestion pipeline.

Responsibilities:
    1. Load every supported document (PDF, TXT) from the documents/ folder,
       preserving per-page metadata (filename, page number).
    2. Split documents into overlapping chunks (RecursiveCharacterTextSplitter).
    3. Generate embeddings for each chunk (GoogleGenerativeAIEmbeddings).
    4. Persist everything into a local Chroma vector database.
    5. Skip regeneration if the documents folder hasn't changed since the
       last run (tracked via a content-hash manifest file).

Run directly to (re)build the vector database:
    python ingest.py
    python ingest.py --force     # force a full rebuild
"""

import argparse
import hashlib
import json
import os
import sys

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from utils import config, logger


# --------------------------------------------------------------------------
# Manifest helpers (used to detect whether documents changed)
#------------------------


def _hash_file(filepath: str) -> str:
    """Return an MD5 hash of a file's contents (fast, good enough for change detection)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_manifest(document_paths: list[str]) -> dict:
    """Build a {filename: hash} map representing the current state of documents/."""
    return {os.path.basename(p): _hash_file(p) for p in document_paths}


def _load_existing_manifest() -> dict | None:
    if not os.path.exists(config.MANIFEST_PATH):
        return None
    try:
        with open(config.MANIFEST_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_manifest(manifest: dict) -> None:
    os.makedirs(config.DB_DIR, exist_ok=True)
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def documents_changed(document_paths: list[str]) -> bool:
    """
    Compare the current documents/ folder against the saved manifest.

    Returns True if documents were added, removed, or modified since the
    last successful ingestion (or if there is no prior manifest / no
    existing Chroma database).
    """
    if not os.path.isdir(config.DB_DIR) or not os.listdir(config.DB_DIR):
        return True

    existing_manifest = _load_existing_manifest()
    if existing_manifest is None:
        return True

    current_manifest = _build_manifest(document_paths)
    return current_manifest != existing_manifest


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def discover_document_paths() -> list[str]:
    """Return full paths of every supported file inside documents/."""
    if not os.path.isdir(config.DOCUMENTS_DIR):
        os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)

    paths = []
    for filename in sorted(os.listdir(config.DOCUMENTS_DIR)):
        if filename.startswith("."):
            continue
        if filename.lower().endswith(config.SUPPORTED_EXTENSIONS):
            paths.append(os.path.join(config.DOCUMENTS_DIR, filename))
    return paths


def load_documents(document_paths: list[str]) -> list[Document]:
    """
    Load all documents into LangChain Document objects.

    Each Document carries metadata: {"source": filename, "page": page_number}.
    Corrupted or unreadable files are skipped with a warning rather than
    crashing the whole ingestion run.
    """
    all_docs: list[Document] = []

    for path in document_paths:
        filename = os.path.basename(path)
        try:
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                # PyPDFLoader already produces one Document per page with
                # a "page" field in metadata (0-indexed) — normalize to 1-indexed
                # for human-friendly citations.
                pages = loader.load()
                for page_doc in pages:
                    page_doc.metadata["source"] = filename
                    raw_page = page_doc.metadata.get("page", 0)
                    page_doc.metadata["page"] = raw_page + 1
                all_docs.extend(pages)

            elif path.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                txt_docs = loader.load()
                for txt_doc in txt_docs:
                    txt_doc.metadata["source"] = filename
                    txt_doc.metadata["page"] = "N/A"
                all_docs.extend(txt_docs)

        except Exception as exc:  # noqa: BLE001 - we want to catch any parser failure
            logger.warning(f"Skipping '{filename}' — could not be read ({exc}).")
            continue

    return all_docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split loaded documents into overlapping chunks.

    Why overlap matters: if we split text into non-overlapping blocks, a
    sentence or fact that straddles the boundary between two chunks can be
    cut in half, and neither chunk alone will contain the full answer.
    An overlap (here, 200 characters) means the tail of one chunk is
    repeated as the head of the next, so answers that span a chunk boundary
    still appear intact in at least one chunk. Larger overlap improves
    recall for boundary-spanning answers at the cost of some redundant
    storage/embedding compute — 200/800 (25%) is a reasonable middle ground.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Try to split on paragraph/sentence boundaries first, only falling
        # back to raw character splits if necessary — keeps chunks coherent.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --------------------------------------------------------------------------
# Embedding + storage
# --------------------------------------------------------------------------

def build_vector_store(chunks: list[Document]) -> Chroma:
    """
    Embed all chunks and persist them into a Chroma vector database.

    Chroma's `from_documents` handles embedding generation internally via
    the provided embedding function; we wrap it with a progress bar for
    user feedback on large document sets by batching manually.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)

    # Wipe any existing collection so we start clean (we already decided
    # to rebuild — see documents_changed()).
    if os.path.isdir(config.DB_DIR):
        for f in os.listdir(config.DB_DIR):
            full = os.path.join(config.DB_DIR, f)
            if os.path.isfile(full) and f != "manifest.json":
                os.remove(full)
            elif os.path.isdir(full):
                import shutil
                shutil.rmtree(full)

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.DB_DIR,
    )

    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks", unit="batch"):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)

    return vector_store


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_ingestion(force: bool = False) -> Chroma:
    """
    Full ingestion pipeline. Returns a ready-to-query Chroma vector store.

    If the documents folder hasn't changed since the last run and a
    database already exists, ingestion is skipped and the existing
    database is loaded instead (unless force=True).
    """
    if not config.GOOGLE_API_KEY:
        logger.error(
            "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
            "and add your Gemini API key before running this program."
        )
        sys.exit(1)

    document_paths = discover_document_paths()

    if not document_paths:
        logger.error(
            f"No supported documents found in '{config.DOCUMENTS_DIR}'. "
            f"Add at least one .pdf or .txt file and try again."
        )
        sys.exit(1)

    if not force and not documents_changed(document_paths):
        logger.info("Documents unchanged since last run — reusing existing vector database.")
        embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        return Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.DB_DIR,
        )

    logger.info("Loading documents...")
    documents = load_documents(document_paths)
    if not documents:
        logger.error("No documents could be successfully loaded (all files may be corrupted).")
        sys.exit(1)
    logger.success(f"Loaded {len(documents)} page(s) from {len(document_paths)} file(s).")

    logger.info("Splitting documents into chunks...")
    chunks = split_documents(documents)
    logger.success(f"Created {len(chunks)} chunk(s) "
                    f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    logger.info("Generating embeddings and saving vector database...")
    try:
        vector_store = build_vector_store(chunks)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to generate embeddings via Gemini API: {exc}")
        sys.exit(1)
    logger.success(f"Vector database saved to '{config.DB_DIR}'.")

    # Save manifest so future runs can detect whether documents changed.
    _save_manifest(_build_manifest(document_paths))

    return vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector database.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if documents haven't changed.",
    )
    args = parser.parse_args()
    run_ingestion(force=args.force)


if __name__ == "__main__":
    main()
