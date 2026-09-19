import re
import logging
from typing import List, Optional
from arxiv_agent.state import AgentState, ParsedSection, AgentStatus

try:
    # pyrefly: ignore [missing-import]
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf  # type: ignore
    except ImportError:
        pymupdf = None  # type: ignore

logger = logging.getLogger(__name__)

# Standard academic section title patterns
SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:\d{1,2}(?:\.\d{1,2})*|\b[I|V|X]+\b|\b[A-Z]\b)\s+)?\s*"
    r"(abstract|introduction|related\s+work|background|preliminaries|"
    r"method(?:ology)?|model(?:\s+architecture)?|approach|proposed\s+(?:method|framework|approach)|"
    r"experiments?(?:\s+and\s+results)?|experimental\s+setup|evaluation|empirical\s+results|"
    r"results(?:\s+and\s+discussion)?|discussion|ablation\s+studies?|"
    r"limitations?(?:\s*(?:and|,)\s*(?:future\s+work|broader\s+impacts?|societal\s+impact|discussion))?|"
    r"broader\s+impacts?|ethics\s+statement|"
    r"conclusions?(?:\s+and\s+future\s+work)?|acknowledgements?|"
    r"references|bibliography|appendix(?:\s+[A-Z0-9])?)\s*$",
    re.IGNORECASE,
)


def clean_pdf_text(text: str) -> str:
    """Fix hyphenated line breaks, ligatures, and formatting artifacts."""
    # Fix hyphenation across lines: e.g. "transfor-\nmer" -> "transformer"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Fix broken newlines within sentences
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class PDFParserNode:
    """Extracts structured sections from research paper PDFs with graceful degradation."""

    def __init__(self):
        pass

    def _parse_with_pymupdf(self, pdf_path: str) -> List[ParsedSection]:
        """Extract sections from PDF using PyMuPDF."""
        if pymupdf is None:
            raise ImportError("PyMuPDF is not installed. Please install `pymupdf`.")
        doc = pymupdf.open(pdf_path)
        sections: List[ParsedSection] = []
        current_heading = "Header & Metadata"
        current_lines: List[str] = []
        current_page_start = 1

        for page_num in range(len(doc)):
            page = doc[page_num]
            current_page_idx = page_num + 1
            blocks = page.get_text("blocks")

            for block in blocks:
                block_text = block[4].strip() if len(block) > 4 else ""
                if not block_text:
                    continue

                lines = [line.strip() for line in block_text.split("\n") if line.strip()]
                for line in lines:
                    match = SECTION_HEADER_REGEX.match(line)
                    # Check if line looks like a major section header
                    if match and len(line) < 60:
                        # Flush previous section
                        if current_lines:
                            content_str = clean_pdf_text("\n".join(current_lines))
                            if len(content_str) > 30:
                                sections.append(
                                    ParsedSection(
                                        heading=current_heading,
                                        content=content_str,
                                        page_start=current_page_start,
                                        page_end=current_page_idx,
                                    )
                                )
                        current_heading = line
                        current_lines = []
                        current_page_start = current_page_idx
                    else:
                        current_lines.append(line)

        # Flush final section
        if current_lines:
            content_str = clean_pdf_text("\n".join(current_lines))
            if len(content_str) > 30:
                sections.append(
                    ParsedSection(
                        heading=current_heading,
                        content=content_str,
                        page_start=current_page_start,
                        page_end=len(doc),
                    )
                )

        doc.close()
        return sections

    def _fallback_from_metadata(self, state: AgentState) -> List[ParsedSection]:
        """Gracefully create structured sections from metadata & abstract when PDF is unavailable or unparseable."""
        paper = state.selected_paper
        if not paper:
            return []

        sections = [
            ParsedSection(
                heading="Abstract & Overview",
                content=f"Paper Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\narXiv ID: {paper.arxiv_id} (Published: {paper.published})\n\nAbstract:\n{paper.abstract}",
                page_start=1,
                page_end=1,
            ),
            ParsedSection(
                heading="Subject Scope & Categories",
                content=f"Categories: {', '.join(paper.categories)}\nEntry URL: {paper.entry_url}\nPDF Link: {paper.pdf_url}\nAuthor Comments: {paper.comment or 'None provided.'}",
                page_start=1,
                page_end=1,
            ),
        ]
        return sections

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.PARSING_PDF
        state.add_log("Stage 4b: Parsing PDF sections...")

        sections: List[ParsedSection] = []

        if state.pdf_path:
            try:
                sections = self._parse_with_pymupdf(state.pdf_path)
            except Exception as e:
                logger.warning(f"PyMuPDF parsing failed ({e}). Triggering fallback.")
                state.add_warning(f"PDF extraction error: {e}. Degrading to abstract/metadata mode.")

        # If parsed sections are empty (e.g. scanned PDF with no text layer or download failure)
        total_extracted_chars = sum(len(s.content) for s in sections)
        if not sections or total_extracted_chars < 100:
            logger.info("Using metadata fallback for paper sections.")
            sections = self._fallback_from_metadata(state)
            state.is_fallback_mode = True
            state.add_warning(
                "PDF text layer was empty or unavailable. System is operating in abstract/metadata mode."
            )

        state.parsed_sections = sections
        state.full_text = "\n\n".join([f"=== {s.heading} ===\n{s.content}" for s in sections])
        state.add_log(
            f"Successfully parsed {len(sections)} section(s) "
            f"({len(state.full_text)} total characters)."
        )
        return state
