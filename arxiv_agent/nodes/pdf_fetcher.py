"""Node 4a: PDF Fetching, Caching, and Validation."""

import os
import re
import logging
from pathlib import Path
from typing import Optional
import requests
from arxiv_agent.config import Config
from arxiv_agent.state import AgentState, AgentStatus

logger = logging.getLogger(__name__)


class PDFFetcherNode:
    """Downloads arXiv research paper PDFs to local cache with validation."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = 30):
        self.cache_dir = cache_dir or (Config.CACHE_DIR / "pdfs")
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, arxiv_id: str) -> str:
        """Sanitize arXiv ID to a safe local file name."""
        clean = re.sub(r"[^\w\.\-]", "_", arxiv_id)
        return f"{clean}.pdf"

    def fetch_pdf(self, arxiv_id: str, pdf_url: str) -> Optional[str]:
        """
        Download PDF if not cached. Returns local file path or None on failure.
        """
        filename = self._sanitize_filename(arxiv_id)
        target_path = self.cache_dir / filename

        # 1. Return cached PDF if already exists and is non-empty
        if target_path.exists() and target_path.stat().st_size > 1024:
            logger.info(f"Using cached PDF: {target_path}")
            return str(target_path.resolve())

        # 2. Download from arXiv PDF URL
        logger.info(f"Downloading PDF from {pdf_url} to {target_path}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) arXiv-Paper-Agent/1.0",
            "Accept": "application/pdf,*/*",
        }

        try:
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=self.timeout)
            response.raise_for_status()

            # Verify content type or start of stream
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            # Validate PDF magic header
            with open(target_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF-"):
                    logger.warning(f"Downloaded file does not have PDF header: {header}")
                    target_path.unlink(missing_ok=True)
                    return None

            logger.info(f"Successfully downloaded PDF ({target_path.stat().st_size} bytes)")
            return str(target_path.resolve())

        except Exception as e:
            logger.warning(f"Failed to download PDF for {arxiv_id} from {pdf_url}: {e}")
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            return None

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.FETCHING_PDF
        state.add_log("Stage 4a: Fetching paper PDF...")

        if not state.selected_paper:
            state.add_error("No paper selected to fetch PDF for.")
            state.status = AgentStatus.FAILED
            return state

        paper = state.selected_paper
        pdf_path = self.fetch_pdf(paper.arxiv_id, paper.pdf_url)

        if pdf_path:
            state.pdf_path = pdf_path
            state.add_log(f"PDF acquired: {pdf_path}")
        else:
            state.pdf_path = None
            warning_msg = (
                f"Could not download full PDF for arXiv:{paper.arxiv_id}. "
                "Agent will fall back to rich abstract & metadata parsing."
            )
            state.add_warning(warning_msg)
            state.add_log(warning_msg)

        return state
