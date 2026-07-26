"""
Retrieval-augmented answer generation.

Given a user question and an existing Chroma vector store, this module:
    1. Retrieves the top-k most similar chunks (similarity search).
    2. Builds a grounded prompt (system instructions + context + question).
    3. Calls the Gemini chat model to generate an answer.
    4. Returns the answer alongside deduplicated source citations.
"""

from dataclasses import dataclass, field
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils import config, prompts, logger


@dataclass
class RetrievedChunk:
    """A single chunk returned from similarity search, with its score."""
    text: str
    source: str
    page: str | int
    similarity_score: float


@dataclass
class RagResponse:
    """The final result returned to the CLI (or any other frontend)."""
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    is_unanswerable: bool = False

    def formatted_sources(self) -> str:
        """
        Return a deduplicated, human-readable list of "file (Page N)" citations,
        in the order the chunks were retrieved.
        """
        if self.is_unanswerable or not self.retrieved_chunks:
            return "(none)"

        seen = set()
        lines = []
        for chunk in self.retrieved_chunks:
            label = f"{chunk.source} (Page {chunk.page})"
            if label not in seen:
                seen.add(label)
                lines.append(label)
        return "\n".join(lines)


def retrieve_chunks(vector_store: Chroma, question: str, k: int = config.RETRIEVAL_K) -> list[RetrievedChunk]:
    """
    Run similarity search against the vector store and return the top-k chunks.

    Uses `similarity_search_with_relevance_scores` so we can optionally
    display how confident the retrieval was (bonus feature).
    """
    try:
        results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Similarity search failed: {exc}")
        return []

    chunks = []
    for doc, score in results:
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", "N/A"),
                similarity_score=score,
            )
        )
    return chunks


def generate_answer(question: str, vector_store: Chroma, k: int = config.RETRIEVAL_K) -> RagResponse:
    """
    Full retrieve-then-generate pipeline for a single question.

    Handles the "no relevant chunks found" case and Gemini API errors
    gracefully, always returning a RagResponse rather than raising.
    """
    logger.info("Retrieving relevant chunks...")
    retrieved = retrieve_chunks(vector_store, question, k=k)

    if not retrieved:
        return RagResponse(answer=prompts.NO_ANSWER_MESSAGE, retrieved_chunks=[], is_unanswerable=True)

    if config.SHOW_SIMILARITY_SCORES:
        for c in retrieved:
            logger.info(f"  - {c.source} (Page {c.page}) | similarity: {c.similarity_score:.3f}")

    context_dicts = [{"text": c.text, "source": c.source, "page": c.page} for c in retrieved]
    user_prompt = prompts.build_full_prompt(question, context_dicts)

    logger.info("Generating response...")
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    try:
        response = llm.invoke([
            SystemMessage(content=prompts.SYSTEM_INSTRUCTIONS),
            HumanMessage(content=user_prompt),
        ])
        if isinstance(response.content, str):
            answer_text = response.content.strip()
        elif isinstance(response.content, list):
            answer_text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in response.content
            ).strip()
        else:
            answer_text = str(response.content).strip()
    except Exception as exc:  # noqa: BLE001 - covers GoogleAPIError and other SDK-level failures
        logger.error(f"Gemini API error while generating the answer: {exc}")
        return RagResponse(
            answer="An error occurred while contacting the Gemini API. Please try again.",
            retrieved_chunks=retrieved,
            is_unanswerable=True,
        )

    is_unanswerable = prompts.NO_ANSWER_MESSAGE.lower() in answer_text.lower()

    return RagResponse(
        answer=answer_text,
        retrieved_chunks=retrieved,
        is_unanswerable=is_unanswerable,
    )
