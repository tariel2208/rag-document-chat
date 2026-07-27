

import sys

from colorama import Fore, Style

from utils import config, logger
from ingest import run_ingestion
from query import generate_answer


BANNER = f"""{Fore.CYAN}{Style.BRIGHT}
  Welcome to our service !!!
  RAG Document Chat — ask questions about your documents

{Style.RESET_ALL}"""


def print_welcome()->None:
    print(BANNER)
    print("Type your question and press Enter.")
    print(f"Type {Fore.YELLOW}exit{Style.RESET_ALL} to quit.\n")


def chat_loop(vector_store) -> None:
    """Interactive loop: read a question, answer it, repeat until 'exit'."""
    while True:
        try:
            question = input(f"{Fore.GREEN}Ask a question:{Style.RESET_ALL}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = generate_answer(question, vector_store, k=config.RETRIEVAL_K)

        print()
        logger.answer(response.answer)
        print()
        logger.sources(response.formatted_sources())
        print("\n" + "-" * 60 + "\n")


def main()->None:
    print_welcome();



  
    if not config.GOOGLE_API_KEY:
        logger.error(
            "GOOGLE_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your Gemini API key to the .env file\n"
            "  3. Run this program again."
        )
        sys.exit(1);

    try:
        vector_store = run_ingestion(force=False)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Unexpected error during document ingestion: {exc}")
        sys.exit(1)

    chat_loop(vector_store)


if __name__ == "__main__":
    main()
