# RAG Document Chat

Ask questions about your own PDF (or TXT) documents and get answers that are
grounded **only** in those documents — with source citations, and an honest
"I don't know" when the answer isn't in there.

## Overview

This is a Retrieval-Augmented Generation (RAG) application. Instead of
asking an LLM to answer from its training data (where it might hallucinate
or be out of date), it:

1. Splits your documents into small overlapping chunks.
2. Embeds each chunk into a vector and stores it in a local Chroma database.
3. When you ask a question, finds the most semantically similar chunks.
4. Feeds only those chunks to the LLM and asks it to answer strictly from
   them, citing the source file and page number.

## Features

- 📄 Multi-PDF (and TXT) ingestion with page-level metadata
- ✂️ Configurable recursive chunking with overlap (handles answers that
  span chunk boundaries)
- 🔎 Semantic similarity search (configurable `k`) with optional score display
- 🧠 Strict grounded prompting — no outside knowledge, no guessing
- 🚫 Explicit "I don't know based on the provided documents." fallback
- 📚 Source citations (`filename.pdf (Page N)`) for every answer
- ⚡ Smart re-ingestion — skips re-embedding if documents haven't changed
- 🎨 Colored terminal output and progress bars during embedding
- 🛡️ Friendly error handling for missing keys, empty folders, corrupted
  PDFs, and API failures

## Requirements

- Python 3.11+
- A Google Gemini API key (free tier available at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

## Installation

```bash
# 1. Clone or download this project, then move into it
cd rag-document-chat

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## API Key Setup

```bash
cp .env.example .env
```

Then open `.env` and set your key:

```
GOOGLE_API_KEY=your-real-gemini-key-here
```

You can generate a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Never commit your real `.env` file — it's already excluded via `.gitignore`.

## Usage

1. Drop one or more `.pdf` (or `.txt`) files into the `documents/` folder.
2. Run the app:

```bash
python app.py
```

The first run will ingest your documents (load → chunk → embed → store).
Subsequent runs reuse the existing database automatically unless the
contents of `documents/` change.

You can also run ingestion manually:

```bash
python ingest.py            # ingest only if documents changed
python ingest.py --force    # force a full rebuild
```

### Example session

```
Ask a question:
> How do employees request leave?

Answer:
Employees should submit a leave request through the HR portal.

Sources:
HR_Handbook.pdf (Page 14)
```

If the answer isn't in your documents:

```
Ask a question:
> What is the capital of France?

Answer:
I don't know based on the provided documents.

Sources:
(none)
```

Type `exit` to quit.

## Configuration

All settings can be overridden in `.env` (see `.env.example`):

| Variable                 | Default                 | Description                              |
|---------------------------|--------------------------|-------------------------------------------|
| `LLM_MODEL`               | `gemini-2.0-flash`             | Chat model used to generate answers       |
| `EMBEDDING_MODEL`         | `models/gemini-embedding-001`  | Embedding model                           |
| `LLM_TEMPERATURE`         | `0`                      | Generation randomness (0 = deterministic) |
| `CHUNK_SIZE`               | `800`                    | Characters per chunk                      |
| `CHUNK_OVERLAP`            | `200`                    | Overlap between consecutive chunks        |
| `RETRIEVAL_K`              | `3`                      | Number of chunks retrieved per question   |
| `SHOW_SIMILARITY_SCORES`   | `true`                   | Print retrieval similarity scores         |

## Folder Structure

```
rag-document-chat/
│── app.py              # CLI entry point / chat loop
│── ingest.py            # Load, chunk, embed, and store documents
│── query.py              # Retrieval + grounded answer generation
│── requirements.txt
│── README.md
│── .env.example
│── .gitignore
│
│── documents/            # Put your PDFs / TXT files here
│── db/                    # Persisted Chroma vector database (auto-generated)
│
└── utils/
    ├── config.py         # Central configuration
    ├── logger.py         # Colored console logging
    └── prompts.py         # Prompt templates & hallucination-prevention rules
```

## How Chunking & Overlap Work

Documents are split using `RecursiveCharacterTextSplitter` with
`chunk_size=800` and `chunk_overlap=200`. The overlap matters because
without it, a fact or sentence that happens to fall right on a chunk
boundary would be split in half — and neither resulting chunk would contain
the full answer. By repeating the last 200 characters of a chunk at the
start of the next one, boundary-spanning answers still appear intact in at
least one retrieved chunk.

## Hallucination Prevention

The prompt sent to the model is split into three explicit sections
(system instructions, retrieved context, user question) and instructs the
model to:

- Only use the retrieved context — never outside/training knowledge
- Never guess
- Respond with exactly `"I don't know based on the provided documents."`
  when the answer isn't present

If similarity search returns no relevant chunks at all, the app short-circuits
and returns the "I don't know" message without even calling the LLM.

## Error Handling

The app gracefully handles:

- Missing `GOOGLE_API_KEY`
- Empty `documents/` folder
- Corrupted or unreadable PDF files (skipped with a warning, ingestion continues)
- No matching chunks for a query
- Gemini API errors (rate limits, network issues, etc.)

## Future Improvements

- Web UI (Streamlit/Gradio) instead of CLI
- Support for `.docx` and `.md` files
- Streaming token-by-token responses
- Multi-turn conversational memory (follow-up questions)
- Re-ranking retrieved chunks with a cross-encoder for higher precision
- Per-document access control / multi-user support
- Swap Chroma for a hosted vector DB (Pinecone, Weaviate) for scale
