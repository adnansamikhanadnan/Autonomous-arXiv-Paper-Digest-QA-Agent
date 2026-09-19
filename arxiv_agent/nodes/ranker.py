"""Node 3: Candidate Selection and Semantic Ranking for Topic Searches."""

import json
import logging
from typing import Optional, List
from arxiv_agent.state import AgentState, PaperMetadata, AgentStatus
from arxiv_agent.llm.provider import BaseLLMProvider, clean_json_response

logger = logging.getLogger(__name__)

RANKER_SYSTEM_PROMPT = """You are an expert AI research assistant. Your task is to evaluate a list of candidate arXiv research papers and select the single most relevant and impactful paper matching the user's research topic.

Evaluate based on:
1. Direct alignment with the specific problem and methodology asked in the topic query.
2. Technical significance and depth indicated by the abstract.
3. Currency/recency of the research.

You MUST respond strictly with a valid JSON object in this format:
{
  "selected_arxiv_id": "2401.12345",
  "relevance_score": 0.95,
  "selection_rationale": "Brief 1-2 sentence explanation of why this paper is the best match."
}
"""


class PaperRankerNode:
    """Ranks and selects the best candidate paper from arXiv search results."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider

    def select_paper(self, query: str, candidates: List[PaperMetadata]) -> PaperMetadata:
        """Select the best matching paper using LLM reasoning or fallback heuristics."""
        if not candidates:
            raise ValueError("No candidate papers available to rank.")

        if len(candidates) == 1:
            selected = candidates[0]
            selected.relevance_score = 1.0
            selected.selection_rationale = "Direct lookup match for specific paper."
            return selected

        if not self.llm:
            # Heuristic fallback: pick first candidate (sorted by relevance from arXiv)
            selected = candidates[0]
            selected.relevance_score = 0.9
            selected.selection_rationale = "Top ranked result from arXiv relevance index."
            return selected

        # Format candidates for the LLM
        candidates_text = ""
        for i, paper in enumerate(candidates, 1):
            candidates_text += (
                f"\n[{i}] arXiv ID: {paper.arxiv_id}\n"
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(paper.authors[:4])}\n"
                f"Published: {paper.published} | Categories: {', '.join(paper.categories)}\n"
                f"Abstract: {paper.abstract}\n"
                f"----------------------------------------"
            )

        prompt = (
            f"User Research Topic: {query}\n\n"
            f"Candidate Papers:\n{candidates_text}\n\n"
            f"Select the single best paper and return JSON."
        )

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=RANKER_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.1,
            )
            cleaned = clean_json_response(raw_response)
            data = json.loads(cleaned)

            selected_id = str(data.get("selected_arxiv_id", "")).strip()
            rationale = str(data.get("selection_rationale", "Selected as top semantic match."))
            score = float(data.get("relevance_score", 0.95))

            # Match against candidate IDs
            for paper in candidates:
                if paper.arxiv_id == selected_id or selected_id in paper.arxiv_id or paper.arxiv_id in selected_id:
                    paper.relevance_score = score
                    paper.selection_rationale = rationale
                    return paper

            # Fallback if LLM generated a non-matching ID
            selected = candidates[0]
            selected.relevance_score = score
            selected.selection_rationale = rationale
            return selected

        except Exception as e:
            logger.warning(f"LLM ranking failed ({e}), falling back to top arXiv result.")
            selected = candidates[0]
            selected.relevance_score = 0.85
            selected.selection_rationale = "Selected based on top arXiv API relevance score."
            return selected

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.RANKING_PAPERS
        state.add_log("Stage 3: Selecting and ranking best matching paper...")

        if not state.candidate_papers:
            state.add_error(f"Cannot select paper: No candidate papers found for '{state.user_query}'.")
            state.status = AgentStatus.FAILED
            return state

        selected = self.select_paper(state.cleaned_query, state.candidate_papers)
        state.selected_paper = selected
        state.add_log(
            f"Selected paper '{selected.title}' (arXiv:{selected.arxiv_id}) | "
            f"Rationale: {selected.selection_rationale}"
        )
        return state
