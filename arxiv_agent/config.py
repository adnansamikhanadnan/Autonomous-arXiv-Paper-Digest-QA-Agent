import os
from pathlib import Path
from typing import Optional

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    """Agent runtime configuration settings."""

    # Provider settings: 'gemini', 'groq', 'openai', 'ollama', 'openrouter', 'mock'
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    # Model names
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # arXiv settings
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
    ARXIV_API_BASE_URL: str = "https://export.arxiv.org/api/query"
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Vector store & Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))

    # Caching
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./.cache_arxiv"))

    @classmethod
    def initialize(cls) -> None:
        """Ensure necessary cache directories exist."""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (cls.CACHE_DIR / "pdfs").mkdir(parents=True, exist_ok=True)
        (cls.CACHE_DIR / "summaries").mkdir(parents=True, exist_ok=True)
