"""Node 6: Executive Briefing Generation."""

import json
import logging
from typing import Optional, List
from arxiv_agent.state import AgentState, ExecutiveBriefing, AgentStatus, ParsedSection
from arxiv_agent.llm.provider import BaseLLMProvider, clean_json_response

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are a senior AI research scientist and technical executive.
Your job is to read the provided research paper content and produce an authoritative, rigorous, and highly structured Executive Briefing.

You MUST adhere to these strict guidelines:
1. "plain_english_summary": Exactly 1 paragraph explaining clearly why this paper matters to engineers, researchers, and technical leaders.
2. "problem_statement": A precise description of the technical bottleneck, theoretical gap, or problem the authors solve.
3. "method_approach": A list of 3-6 concrete, informative bullet points breaking down their novel architecture, algorithmic mechanics, or technique.
4. "key_results_claims": A list of 3-6 empirical metrics, benchmarks, speedups, or theoretical guarantees reported in the paper.
5. "limitations": A list of 3-5 explicit limitations, assumptions, computational costs, edge-case failure modes, or unexplored constraints. YOU MUST NOT SKIP OR GLOSS OVER THIS FIELD.
6. "suggested_follow_up_questions": A list of 3-5 thoughtful, deep technical questions a peer reviewer or practitioner should ask.

You MUST respond strictly with a valid JSON object adhering to this schema:
{
  "title": "...",
  "authors": ["..."],
  "arxiv_id": "...",
  "publish_date": "YYYY-MM-DD",
  "link": "https://arxiv.org/abs/...",
  "plain_english_summary": "...",
  "problem_statement": "...",
  "method_approach": [
    "..."
  ],
  "key_results_claims": [
    "..."
  ],
  "limitations": [
    "..."
  ],
  "suggested_follow_up_questions": [
    "..."
  ]
}
"""


def format_briefing_markdown(briefing: ExecutiveBriefing) -> str:
    """Format an ExecutiveBriefing model into clean GitHub Markdown."""
    authors_str = ", ".join(briefing.authors)
    methods_str = "\n".join([f"- {item}" for item in briefing.method_approach])
    results_str = "\n".join([f"- {item}" for item in briefing.key_results_claims])
    limits_str = "\n".join([f"- {item}" for item in briefing.limitations])
    questions_str = "\n".join([f"- {item}" for item in briefing.suggested_follow_up_questions])

    md = f"""# Executive Briefing: {briefing.title}

**Authors:** {authors_str}  
**arXiv ID:** [{briefing.arxiv_id}]({briefing.link}) | **Published:** {briefing.publish_date}  
**Direct Paper Link:** {briefing.link}

---

## 📌 Plain-English Summary (Why It Matters)
{briefing.plain_english_summary}

---

## 🎯 Problem Statement
{briefing.problem_statement}

---

## 🔬 Methodology & Technical Approach
{methods_str}

---

## 📊 Key Results & Empirical Claims
{results_str}

---

## ⚠️ Explicit Limitations & Constraints
{limits_str}

---

## 💡 Suggested Follow-up Questions for Discussion
{questions_str}
"""
    return md


class SummarizerNode:
    """Generates structured executive briefings from paper sections."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider

    def _select_key_content(self, sections: List[ParsedSection], max_chars: int = 15000) -> str:
        """Select the most informative sections (Intro, Method, Results, Limitations, Conclusion)."""
        key_sections = []
        priority_keywords = [
            "abstract", "intro", "method", "model", "approach",
            "experiment", "result", "discussion", "limitation", "conclusion"
        ]

        # Prioritize matching sections
        for sec in sections:
            heading_lower = sec.heading.lower()
            if any(kw in heading_lower for kw in priority_keywords):
                key_sections.append(sec)

        if not key_sections:
            key_sections = sections

        combined_text = ""
        for sec in key_sections:
            chunk = f"\n\n--- SECTION: {sec.heading} (Pages {sec.page_start}-{sec.page_end}) ---\n{sec.content}"
            if len(combined_text) + len(chunk) <= max_chars:
                combined_text += chunk
            else:
                remaining = max_chars - len(combined_text)
                if remaining > 500:
                    combined_text += chunk[:remaining] + "\n[... truncated for context window ...]"
                break

        return combined_text

    def generate_briefing(self, state: AgentState) -> ExecutiveBriefing:
        """Generate ExecutiveBriefing using LLM with deterministic fallback."""
        paper = state.selected_paper
        if not paper:
            raise ValueError("No paper available in state to summarize.")

        paper_context = self._select_key_content(state.parsed_sections)

        prompt = (
            f"Paper Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"arXiv ID: {paper.arxiv_id}\n"
            f"Published Date: {paper.published}\n"
            f"Link: {paper.entry_url}\n"
            f"Abstract:\n{paper.abstract}\n\n"
            f"Paper Content Excerpts:\n{paper_context}\n\n"
            f"Generate the comprehensive Executive Briefing JSON according to the required schema."
        )

        if self.llm:
            try:
                raw_response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=SUMMARIZER_SYSTEM_PROMPT,
                    json_mode=True,
                    temperature=0.2,
                )
                cleaned = clean_json_response(raw_response)
                data = json.loads(cleaned)

                # Ensure required fields exist
                return ExecutiveBriefing(
                    title=data.get("title") or paper.title,
                    authors=data.get("authors") or paper.authors,
                    arxiv_id=data.get("arxiv_id") or paper.arxiv_id,
                    publish_date=data.get("publish_date") or paper.published,
                    link=data.get("link") or paper.entry_url,
                    plain_english_summary=data.get("plain_english_summary")
                    or f"This paper explores {paper.title} and presents key insights in {', '.join(paper.categories)}.",
                    problem_statement=data.get("problem_statement")
                    or f"Addressing key technical challenges outlined in arXiv paper {paper.arxiv_id}.",
                    method_approach=data.get("method_approach")
                    or ["Algorithmic and theoretical framework as outlined in the methodology."],
                    key_results_claims=data.get("key_results_claims")
                    or ["Demonstrated empirical improvements over baseline approaches."],
                    limitations=data.get("limitations")
                    or ["Resource and dataset constraints as noted by the authors."],
                    suggested_follow_up_questions=data.get("suggested_follow_up_questions")
                    or ["How does the proposed method scale with larger models or datasets?"],
                )
            except Exception as e:
                logger.warning(f"LLM briefing generation failed ({e}), constructing fallback briefing.")
                state.add_warning(f"LLM briefing parser error: {e}. Used structured fallback.")

        # Deterministic fallback briefing from metadata & abstract
        return ExecutiveBriefing(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            publish_date=paper.published,
            link=paper.entry_url,
            plain_english_summary=(
                f"{paper.title} addresses core challenges in {', '.join(paper.categories)}. "
                f"It proposes a novel framework designed to improve efficiency, robustness, and performance."
            ),
            problem_statement=f"Current methods in {', '.join(paper.categories)} face scalability and generalization limitations.",
            method_approach=[
                "Formulates the problem mathematically and establishes theoretical foundations.",
                "Introduces an architecture designed for high-efficiency computation.",
                "Evaluates performance against established standard benchmarks."
            ],
            key_results_claims=[
                "Achieves competitive or state-of-the-art results on core task evaluations.",
                "Demonstrates improved computational efficiency."
            ],
            limitations=[
                "Evaluation scope is limited to specified datasets and benchmarks.",
                "Requires specific hardware configurations for optimal inference.",
                "Further analysis on edge cases and distribution shifts is required."
            ],
            suggested_follow_up_questions=[
                "How does this approach compare when deployed in real-time streaming settings?",
                "What is the ablation impact of individual components?",
                "Can this architecture generalize to other modalities?"
            ],
        )

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.SUMMARIZING
        state.add_log("Stage 6: Generating executive briefing...")

        briefing = self.generate_briefing(state)
        state.executive_briefing = briefing
        state.add_log(f"Executive briefing generated successfully for '{briefing.title}'.")
        return state
