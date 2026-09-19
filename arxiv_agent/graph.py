"""Stateful Agent Graph Orchestration for arXiv Paper Digest, Comparative Synthesis & QA."""

import logging
from typing import Optional, Callable
from arxiv_agent.config import Config
from arxiv_agent.state import AgentState, AgentStatus, QAMessage, ComparativeAnalysis
from arxiv_agent.llm.provider import BaseLLMProvider, get_llm_provider
from arxiv_agent.nodes.query_understanding import QueryUnderstandingNode
from arxiv_agent.nodes.arxiv_retrieval import ArxivRetrievalNode
from arxiv_agent.nodes.ranker import PaperRankerNode
from arxiv_agent.nodes.pdf_fetcher import PDFFetcherNode
from arxiv_agent.nodes.pdf_parser import PDFParserNode
from arxiv_agent.nodes.vector_store import VectorStoreNode
from arxiv_agent.nodes.summarizer import SummarizerNode
from arxiv_agent.nodes.qa_engine import QAEngineNode
from arxiv_agent.nodes.comparative_synthesizer import ComparativeSynthesizerNode

logger = logging.getLogger(__name__)


class ArxivAgentGraph:
    """
    Explicit Stateful Agent Graph for arXiv Discovery, Comparative Matrix, Summarization, and Grounded QA.
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        on_step_callback: Optional[Callable[[str, AgentState], None]] = None,
    ):
        Config.initialize()

        self.llm = llm_provider or get_llm_provider()
        self.on_step = on_step_callback

        # Initialize Graph Nodes
        self.query_node = QueryUnderstandingNode(llm_provider=self.llm)
        self.retrieval_node = ArxivRetrievalNode()
        self.ranker_node = PaperRankerNode(llm_provider=self.llm)
        self.fetcher_node = PDFFetcherNode()
        self.parser_node = PDFParserNode()
        self.vector_node = VectorStoreNode(llm_provider=self.llm)
        self.summarizer_node = SummarizerNode(llm_provider=self.llm)
        self.qa_node = QAEngineNode(vector_node=self.vector_node, llm_provider=self.llm)
        self.comparative_node = ComparativeSynthesizerNode(llm_provider=self.llm)

        # Graph State
        self.current_state: Optional[AgentState] = None

    def _notify(self, step_name: str, state: AgentState) -> None:
        """Call step observer callback if provided."""
        if self.on_step:
            try:
                self.on_step(step_name, state)
            except Exception as e:
                logger.error(f"Step callback error: {e}")

    def run(self, user_query: str, generate_comparison: bool = False) -> AgentState:
        """
        Execute full pipeline from query understanding to executive briefing and optional comparative synthesis.
        """
        state = AgentState(user_query=user_query)
        self.current_state = state

        # --- Node 1: Query Understanding ---
        self._notify("query_understanding_start", state)
        state = self.query_node.process(state)
        self._notify("query_understanding_end", state)

        # --- Node 2: arXiv Retrieval ---
        self._notify("arxiv_retrieval_start", state)
        state = self.retrieval_node.process(state)
        self._notify("arxiv_retrieval_end", state)

        # Failure Check: No papers found
        if not state.candidate_papers:
            state.status = AgentStatus.FAILED
            state.add_error(
                f"No papers found on arXiv for query '{user_query}'. "
                "Try using broader keywords or a specific arXiv ID (e.g. 2401.12345)."
            )
            self._notify("pipeline_failed", state)
            return state

        # Optional: Generate multi-paper comparative synthesis if requested and multiple candidates exist
        if generate_comparison and len(state.candidate_papers) > 1:
            self._notify("comparison_start", state)
            state = self.comparative_node.process(state)
            self._notify("comparison_end", state)

        # --- Node 3: Selection & Ranking (Conditional Edge) ---
        self._notify("ranking_start", state)
        state = self.ranker_node.process(state)
        self._notify("ranking_end", state)

        if not state.selected_paper or state.status == AgentStatus.FAILED:
            return state

        # --- Node 4a: PDF Fetching ---
        self._notify("pdf_fetch_start", state)
        state = self.fetcher_node.process(state)
        self._notify("pdf_fetch_end", state)

        # --- Node 4b: PDF Parsing (with abstract fallback) ---
        self._notify("pdf_parse_start", state)
        state = self.parser_node.process(state)
        self._notify("pdf_parse_end", state)

        # --- Node 5: Chunk & Embed Vector Index (Parent-Child RRF) ---
        self._notify("vector_indexing_start", state)
        state = self.vector_node.process(state)
        self._notify("vector_indexing_end", state)

        # --- Node 6: Executive Briefing Generation ---
        self._notify("summarizing_start", state)
        state = self.summarizer_node.process(state)
        self._notify("summarizing_end", state)

        # --- Node 7: Setup QA Loop ---
        self._notify("qa_setup_start", state)
        state = self.qa_node.process(state)
        self._notify("qa_setup_end", state)

        self.current_state = state
        return state

    def ask(self, question: str) -> QAMessage:
        """
        Ask a follow-up question in the interactive QA loop.
        """
        if not self.current_state or not self.current_state.selected_paper:
            raise RuntimeError("Agent must run a paper query before entering the QA loop.")

        return self.qa_node.answer_question(question, self.current_state)

    def compare_candidates(self) -> ComparativeAnalysis:
        """
        Trigger comparative analysis across candidate papers.
        """
        if not self.current_state or not self.current_state.candidate_papers:
            raise RuntimeError("No candidate papers in state to compare.")
        return self.comparative_node.synthesize(
            self.current_state.user_query, self.current_state.candidate_papers
        )
