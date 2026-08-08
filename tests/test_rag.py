from unittest.mock import patch

from query import RetrievedChunk, is_relevant, generate_answer
from utils import config,prompts

def test_unswarable_question():
    chunks = [
        RetrievedChunk(
            text = "My name is Tariel",
            source ="file1.txt",
            page = "N/A",
            similarity_score=0.554,
        )
    ]

    assert is_relevant(chunks) is True 


def test_unanswerable_question():
    chunks = [
        RetrievedChunk(
            text="Some unrelated content or information",
            source = "file1.txt",
            page = "N/A",
            similarity_score=0.438,
        )
    ]

    assert is_relevant(chunks) is False

def test_unanswerable_question_does_not_call_gemini():
    fake_chunks = [ 
        RetrievedChunk(
            text = "Some unrelated content",
            source = "file1.txt",
            page = "N/A",
            similarity_score=0.438,
        )
    ]
    fake_vector_store = object()

    with patch(
        "query.retrieve_chunks",
        return_value = fake_chunks
    ), patch(
        "query.create_llm"
    ) as mock_llm:
        response = generate_answer(
            "Who is Nikola Tesla",
            fake_vector_store
        )

        assert response.is_unanswerable is True
        assert response.answer == prompts.NO_ANSWER_MESSAGE

        # Gemini must NOT be called
        mock_llm.assert_not_called()