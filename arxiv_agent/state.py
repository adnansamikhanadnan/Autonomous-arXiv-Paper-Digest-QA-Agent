"""State definitions and Pydantic data schemas for the arXiv Agent."""

from enum import Enum
from typing import List, Dict, Optional, Any
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """User query classification type."""
    TOPIC_SEARCH = "topic_search"
    SPECIFIC_PAPER = "specific_paper"
    UNKNOWN = "unknown"


class PaperMetadata(BaseModel):
    """Metadata representing an arXiv research paper."""
    arxiv_id: str = Field(description="arXiv identifier, e.g., '2401.12345'")
    title: str = Field(description="Title of the research paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(description="Full text abstract")
    published: str = Field(description="Publication or submission date (YYYY-MM-DD)")
    pdf_url: str = Field(description="Direct URL to the arXiv PDF")
    entry_url: str = Field(description="arXiv abstract landing page URL")
    categories: List[str] = Field(default_factory=list, description="Primary and secondary arXiv category tags")
    comment: Optional[str] = Field(default=None, description="Author comments, e.g., pages, conference name")
    relevance_score: Optional[float] = Field(default=None, description="Ranking score for topic searches")
    selection_rationale: Optional[str] = Field(default=None, description="Explanation why this paper was selected")


class ParsedSection(BaseModel):
    """A structured section extracted from the research paper PDF."""
    heading: str = Field(description="Heading or section name, e.g., 'Introduction', 'Method'")
    content: str = Field(description="Body text belonging to this section")
    page_start: int = Field(default=1, description="Starting 1-indexed page number")
    page_end: int = Field(default=1, description="Ending 1-indexed page number")


class ChunkRecord(BaseModel):
    """A chunk of text indexed in the vector store with parent-child hierarchical metadata."""
    chunk_id: str
    parent_id: Optional[str] = None
    section_title: str
    page_number: int
    text: str                          # Child micro-chunk text for precise vector/keyword retrieval
    parent_text: Optional[str] = None  # Full parent chunk text for rich contextual generation
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None


class ExecutiveBriefing(BaseModel):
    """Structured executive briefing schema required by assessment."""
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="Paper authors")
    arxiv_id: str = Field(description="arXiv identifier")
    publish_date: str = Field(description="Publish date")
    link: str = Field(description="URL to the paper")
    plain_english_summary: str = Field(
        description="1-paragraph plain-English summary explaining why this paper matters"
    )
    problem_statement: str = Field(
        description="The core problem the authors are attempting to solve"
    )
    method_approach: List[str] = Field(
        description="Bullet points explaining the methodology, architecture, or theoretical framework"
    )
    key_results_claims: List[str] = Field(
        description="Key results, performance numbers, benchmarks, or empirical findings"
    )
    limitations: List[str] = Field(
        description="Explicit limitations, assumptions, computational overhead, or failure modes (MANDATORY)"
    )
    suggested_follow_up_questions: List[str] = Field(
        description="Suggested follow-up questions a reader might ask"
    )


class QAMessage(BaseModel):
    """A message in the conversational QA loop with quantitative grounding telemetry."""
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Text message content")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved chunk citations supporting the assistant answer"
    )
    grounded: bool = Field(
        default=True,
        description="Flag indicating whether answer is strictly grounded in retrieved chunks"
    )
    grounding_confidence: float = Field(
        default=1.0,
        description="Quantitative factual grounding confidence score (0.0 to 1.0)"
    )
    grounding_level: str = Field(
        default="HIGH",
        description="Grounding tier: 'HIGH', 'MEDIUM', or 'REFUSAL'"
    )


class PaperComparisonRow(BaseModel):
    """A single paper entry in a comparative synthesis matrix."""
    arxiv_id: str
    title: str
    core_approach: str
    key_advantages: str
    primary_limitations: str


class ComparativeAnalysis(BaseModel):
    """Multi-paper comparative synthesis artifact comparing top candidate papers."""
    topic: str
    papers: List[PaperComparisonRow] = Field(default_factory=list)
    synthesis_summary: str = Field(description="High-level synthesis of tradeoffs across papers")
    recommended_reading_order: List[str] = Field(default_factory=list)


class AgentStatus(str, Enum):
    """State progression status."""
    IDLE = "idle"
    UNDERSTANDING_QUERY = "understanding_query"
    RETRIEVING_METADATA = "retrieving_metadata"
    RANKING_PAPERS = "ranking_papers"
    FETCHING_PDF = "fetching_pdf"
    PARSING_PDF = "parsing_pdf"
    INDEXING_CHUNKS = "indexing_chunks"
    SUMMARIZING = "summarizing"
    COMPARING = "comparing"
    READY_FOR_QA = "ready_for_qa"
    FAILED = "failed"


class AgentState(BaseModel):
    """The complete shared state passed across graph nodes."""
    user_query: str
    intent: IntentType = IntentType.UNKNOWN
    cleaned_query: str = ""
    extracted_arxiv_id: Optional[str] = None
    
    # Retrieval & Selection
    candidate_papers: List[PaperMetadata] = Field(default_factory=list)
    selected_paper: Optional[PaperMetadata] = None
    
    # Parsing & Text
    pdf_path: Optional[str] = None
    parsed_sections: List[ParsedSection] = Field(default_factory=list)
    full_text: str = ""
    is_fallback_mode: bool = False  # True if abstract was used because PDF parsing failed
    
    # Vector store (Parent-Child index)
    chunks: List[ChunkRecord] = Field(default_factory=list)
    
    # Outputs
    executive_briefing: Optional[ExecutiveBriefing] = None
    comparative_analysis: Optional[ComparativeAnalysis] = None
    qa_history: List[QAMessage] = Field(default_factory=list)
    
    # Metadata & Tracking
    status: AgentStatus = AgentStatus.IDLE
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)

    def add_log(self, message: str) -> None:
        """Append log trace."""
        self.logs.append(message)

    def add_warning(self, warning: str) -> None:
        """Append warning."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Append error."""
        self.errors.append(error)
