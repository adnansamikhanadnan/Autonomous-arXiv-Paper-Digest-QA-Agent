"""Node 1: Query Understanding & Intent Parsing."""

import re
import logging
from typing import Optional
from arxiv_agent.state import AgentState, IntentType, AgentStatus
from arxiv_agent.llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)

# Regular expressions for arXiv ID formats:
# Standard modern: YYMM.NNNNN(vN) e.g., 2401.12345, 2305.18290v2
# Old-style: archive/YYMMNNN or subject.class/YYMMNNN e.g., cs/0601001, math.GT/0309136, hep-th/9912012
MODERN_ARXIV_PATTERN = r"(\b\d{4}\.\d{4,5}(?:v\d+)?\b)"
OLD_ARXIV_PATTERN = r"(\b[a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}(?:v\d+)?\b)"
URL_ARXIV_PATTERN = r"(?:arxiv\.org/(?:abs|pdf)/)([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)"


class QueryUnderstandingNode:
    """Parses user input to identify intent (specific paper vs topic search) and extracts query tokens or arXiv ID."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider

    @staticmethod
    def extract_arxiv_id(text: str) -> Optional[str]:
        """Extract sanitized arXiv ID from URL, prefix, or raw ID."""
        text = text.strip()

        # Check URL pattern first
        url_match = re.search(URL_ARXIV_PATTERN, text, re.IGNORECASE)
        if url_match:
            return url_match.group(1).rstrip(".pdf")

        # Check explicit prefix (e.g. arXiv:2401.12345)
        prefix_match = re.search(r"arxiv:\s*([\w\.\-/]+)", text, re.IGNORECASE)
        if prefix_match:
            candidate = prefix_match.group(1).rstrip(".pdf")
            if re.match(MODERN_ARXIV_PATTERN, candidate) or re.match(OLD_ARXIV_PATTERN, candidate):
                return candidate

        # Check modern ID pattern
        mod_match = re.search(MODERN_ARXIV_PATTERN, text)
        if mod_match:
            return mod_match.group(1)

        # Check old-style ID pattern
        old_match = re.search(OLD_ARXIV_PATTERN, text)
        if old_match:
            return old_match.group(1)

        return None

    @staticmethod
    def clean_topic_query(query: str) -> str:
        """Clean natural language query for arXiv search API."""
        cleaned = query.strip()
        # Iteratively strip conversational preamble
        pattern = r"^(?:can you\s+)?(?:please\s+)?(?:summarize|tell me about|find|search for|look up|get)?\s*(?:papers?\s+on|recent\s+work\s+on|work\s+on)?\s*"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        # Remove trailing punctuation
        cleaned = re.sub(r"[?!.,;]+$", "", cleaned).strip()
        return cleaned if cleaned else query.strip()

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.UNDERSTANDING_QUERY
        raw_query = state.user_query.strip()
        state.add_log(f"Stage 1: Parsing user query: '{raw_query}'")

        arxiv_id = self.extract_arxiv_id(raw_query)

        if arxiv_id:
            state.intent = IntentType.SPECIFIC_PAPER
            state.extracted_arxiv_id = arxiv_id
            state.cleaned_query = arxiv_id
            state.add_log(f"Detected SPECIFIC_PAPER intent with arXiv ID: {arxiv_id}")
        else:
            state.intent = IntentType.TOPIC_SEARCH
            state.extracted_arxiv_id = None
            state.cleaned_query = self.clean_topic_query(raw_query)
            state.add_log(f"Detected TOPIC_SEARCH intent with cleaned query: '{state.cleaned_query}'")

        return state
