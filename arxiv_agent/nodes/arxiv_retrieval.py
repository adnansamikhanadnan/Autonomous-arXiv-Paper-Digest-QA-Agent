"""Node 2: arXiv API Retrieval & Metadata Parsing."""

import re
import urllib.parse
import xml.etree.ElementTree as ET
import logging
from typing import List, Optional
import requests
from arxiv_agent.config import Config
from arxiv_agent.state import AgentState, PaperMetadata, IntentType, AgentStatus

logger = logging.getLogger(__name__)

# Atom XML Namespaces used by arXiv API
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and clean newlines in metadata strings."""
    if not text:
        return ""
    # Replace multiple whitespaces and newlines with a single space
    return re.sub(r"\s+", " ", text).strip()


class ArxivRetrievalNode:
    """Queries official arXiv Atom API to fetch candidate paper metadata."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url or Config.ARXIV_API_BASE_URL
        self.timeout = timeout

    def _parse_entry(self, entry: ET.Element) -> Optional[PaperMetadata]:
        """Parse an individual Atom <entry> XML element into a PaperMetadata model."""
        id_elem = entry.find("atom:id", ATOM_NS)
        if id_elem is None or not id_elem.text:
            return None

        # Extract clean arXiv ID (e.g. http://arxiv.org/abs/2401.12345v1 -> 2401.12345)
        raw_id = id_elem.text.strip()
        id_match = re.search(r"arxiv\.org/abs/([^/]+)$", raw_id)
        arxiv_id = id_match.group(1) if id_match else raw_id

        # Clean title
        title_elem = entry.find("atom:title", ATOM_NS)
        title = clean_text(title_elem.text if title_elem is not None else "Untitled Paper")

        # Authors
        authors = []
        for author in entry.findall("atom:author", ATOM_NS):
            name_elem = author.find("atom:name", ATOM_NS)
            if name_elem is not None and name_elem.text:
                authors.append(clean_text(name_elem.text))

        # Abstract
        summary_elem = entry.find("atom:summary", ATOM_NS)
        abstract = clean_text(summary_elem.text if summary_elem is not None else "")

        # Published date
        pub_elem = entry.find("atom:published", ATOM_NS)
        pub_date = pub_elem.text[:10] if pub_elem is not None and pub_elem.text else "Unknown"

        # Categories
        categories = []
        for cat in entry.findall("atom:category", ATOM_NS):
            term = cat.get("term")
            if term:
                categories.append(term)

        # PDF & Entry URLs
        clean_id_base = re.sub(r"v\d+$", "", arxiv_id)
        pdf_url = f"https://arxiv.org/pdf/{clean_id_base}.pdf"
        entry_url = f"https://arxiv.org/abs/{clean_id_base}"

        # Find explicit pdf link if present in entry links
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", pdf_url)

        # Comments
        comment_elem = entry.find("arxiv:comment", ATOM_NS)
        comment = clean_text(comment_elem.text) if comment_elem is not None else None

        return PaperMetadata(
            arxiv_id=clean_id_base,
            title=title,
            authors=authors,
            abstract=abstract,
            published=pub_date,
            pdf_url=pdf_url,
            entry_url=entry_url,
            categories=categories,
            comment=comment,
        )

    def fetch_by_id(self, arxiv_id: str) -> List[PaperMetadata]:
        """Fetch metadata for a specific arXiv ID."""
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        params = {"id_list": clean_id, "max_results": 1}
        return self._execute_query(params)

    def search_by_topic(self, topic: str, max_results: int = 5) -> List[PaperMetadata]:
        """Search arXiv by topic / keyword query."""
        clean_topic = topic.strip()
        # Build search query: search across all fields (title, abstract, authors)
        search_query = f"all:{clean_topic}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        results = self._execute_query(params)

        # If 0 results, try relaxing the query (strip special chars)
        if not results and " " in clean_topic:
            simplified = " ".join([w for w in clean_topic.split() if len(w) > 3])
            if simplified and simplified != clean_topic:
                params["search_query"] = f"all:{simplified}"
                results = self._execute_query(params)

        return results

    def _execute_query(self, params: dict) -> List[PaperMetadata]:
        """Execute HTTP request to arXiv API and parse XML response."""
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        logger.info(f"Querying arXiv API: {url}")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            papers: List[PaperMetadata] = []
            for entry in root.findall("atom:entry", ATOM_NS):
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
            return papers
        except Exception as e:
            logger.error(f"arXiv API request failed: {e}")
            return []

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.RETRIEVING_METADATA
        state.add_log("Stage 2: Calling arXiv API to retrieve paper metadata...")

        if state.intent == IntentType.SPECIFIC_PAPER and state.extracted_arxiv_id:
            papers = self.fetch_by_id(state.extracted_arxiv_id)
        else:
            papers = self.search_by_topic(state.cleaned_query, max_results=Config.ARXIV_MAX_RESULTS)

        state.candidate_papers = papers

        if not papers:
            warning_msg = f"arXiv returned zero candidate papers for query: '{state.user_query}'"
            state.add_warning(warning_msg)
            state.add_log(warning_msg)
        else:
            state.add_log(f"Successfully retrieved {len(papers)} candidate paper(s) from arXiv.")

        return state
