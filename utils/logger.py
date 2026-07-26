"""
logger.py
---------
Lightweight colored console logging.

We avoid pulling in a heavy logging framework since this is a small CLI
app. `colorama` gives us cross-platform ANSI colors (including Windows).
"""

from colorama import Fore, Style, init

# Enables ANSI color codes on Windows terminals too; safe no-op elsewhere.
init(autoreset=True)


def info(message: str) -> None:
    """Neutral, informational log line (e.g. progress updates)."""
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {message}")


def success(message: str) -> None:
    """Successful completion of a step."""
    print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} {message}")


def warning(message: str) -> None:
    """Something unexpected but not fatal."""
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {message}")


def error(message: str) -> None:
    """A failure the user needs to address."""
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")


def answer(message: str) -> None:
    """Highlight the model's answer."""
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Answer:{Style.RESET_ALL}\n{message}")


def sources(message: str) -> None:
    """Highlight the source citations."""
    print(f"{Fore.BLUE}{Style.BRIGHT}Sources:{Style.RESET_ALL}\n{message}")
