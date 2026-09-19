"""Node 7: Grounded RAG Question-Answering Engine with Quantitative Grounding Telemetry."""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from arxiv_agent.config import Config
from arxiv_agent.state import AgentState, QAMessage, AgentStatus
from arxiv_agent.nodes.vector_store import VectorStoreNode
from arxiv_agent.llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are a rigorous scientific QA assistant grounded strictly in the provided research paper excerpts.

CRITICAL GROUNDING RULES:
1. You MUST answer the user's question relying ONLY on the retrieved paper excerpts provided below.
2. For EVERY factual claim or number you mention, cite the corresponding section and page number in brackets, e.g., [Methodology, Page 3] or [Results, Page 5].
3. ANTI-HALLUCINATION REQUIREMENT: If the retrieved excerpts do NOT contain sufficient information to answer the question accurately, you MUST explicitly state:
   "Based on the retrieved sections of this paper, there is not enough information to answer this question."
   NEVER invent methodologies, hyperparameters, performance numbers, or claims not present in the context.
4. Keep answers concise, factual, and scientifically precise.
"""


def calculate_grounding_score(answer: str, context: str) -> float:
    """
    Calculate factual grounding overlap score (0.0 to 1.0) between answer claims and source context.
    """
    if not answer or not context:
        return 0.0

    # If answer is a refusal
    refusal_keywords = ["not contain information", "not enough information", "does not mention", "not discussed in"]
    if any(kw in answer.lower() for kw in refusal_keywords):
        return 0.0

    # Extract substantive words
    answer_words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", answer.lower())
    context_words: Set[str] = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", context.lower()))

    # Ignore generic conversational stop words
    stop_words = {"based", "section", "page", "table", "figure", "paper", "author", "propose", "method", "this", "that", "with", "from", "their", "which", "also", "have", "were"}
    filtered_answer = [w for w in answer_words if w not in stop_words]

    if not filtered_answer:
        return 0.8

    matches = sum(1 for w in filtered_answer if w in context_words)
    overlap_ratio = matches / len(filtered_answer)

    # Scale to confidence (0.5 to 1.0 range for grounded content)
    return min(1.0, max(0.0, round(0.4 + (overlap_ratio * 0.6), 2)))


class QAEngineNode:
    """Handles grounded question-answering over indexed paper chunks with grounding telemetry."""

    def __init__(
        self,
        vector_node: VectorStoreNode,
        llm_provider: Optional[BaseLLMProvider] = None,
        top_k: int = 5,
    ):
        self.vector_node = vector_node
        self.llm = llm_provider
        self.top_k = top_k or Config.TOP_K_CHUNKS

    def answer_question(self, question: str, state: AgentState) -> QAMessage:
        """
        Execute grounded RAG query: retrieve top-k chunks, format context with citations, and generate answer.
        """
        if not question or not question.strip():
            return QAMessage(
                role="assistant",
                content="Please provide a valid question about the paper.",
                citations=[],
                grounded=True,
                grounding_confidence=1.0,
                grounding_level="HIGH",
            )

        # 1. Retrieve top-k chunks using RRF vector search
        scored_chunks = self.vector_node.index.query(question, top_k=self.top_k)

        # Build context string using rich parent_text when available
        context_parts = []
        citations_data = []
        full_context_text = ""

        for chunk, score in scored_chunks:
            source_tag = f"Section: \"{chunk.section_title}\" (Page {chunk.page_number})"
            content_to_use = chunk.parent_text if chunk.parent_text else chunk.text
            context_parts.append(f"[{source_tag} | Chunk: {chunk.chunk_id}]\n{content_to_use}")
            full_context_text += " " + content_to_use
            citations_data.append({
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_title,
                "page": chunk.page_number,
                "score": round(score, 4),
                "preview": chunk.text[:120] + "...",
            })

        combined_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

        # Format multi-turn conversation history
        history_context = ""
        if state.qa_history:
            recent_turns = state.qa_history[-6:]
            formatted_turns = []
            for msg in recent_turns:
                formatted_turns.append(f"{msg.role.capitalize()}: {msg.content}")
            history_context = "Recent Conversation History:\n" + "\n".join(formatted_turns) + "\n\n"

        paper_title = state.selected_paper.title if state.selected_paper else "Research Paper"
        arxiv_id = state.selected_paper.arxiv_id if state.selected_paper else "N/A"

        prompt = (
            f"Target Paper: {paper_title} (arXiv:{arxiv_id})\n\n"
            f"{history_context}"
            f"Retrieved Excerpts from Paper:\n{combined_context}\n\n"
            f"User Question: {question}\n\n"
            f"Answer the user's question accurately with citations to section and page numbers."
        )

        if not self.llm:
            first_sec = citations_data[0]["section"] if citations_data else "General"
            first_page = citations_data[0]["page"] if citations_data else 1
            answer_text = (
                f"Based on [{first_sec}, Page {first_page}], the paper discusses this topic in the context "
                f"of {paper_title}. Relevant context excerpt: \"{citations_data[0]['preview'] if citations_data else 'N/A'}\""
            )
            grounded = True
        else:
            try:
                answer_text = self.llm.generate(
                    prompt=prompt,
                    system_prompt=QA_SYSTEM_PROMPT,
                    temperature=0.1,
                )
                grounded = not any(
                    phrase in answer_text.lower()
                    for phrase in [
                        "not contain information",
                        "not enough information",
                        "does not mention",
                        "not discussed in",
                        "unsupported by",
                    ]
                )
            except Exception as e:
                logger.error(f"QA LLM generation error: {e}")
                answer_text = f"An error occurred while generating the answer: {e}"
                grounded = False

        # Calculate factual grounding confidence score
        confidence = calculate_grounding_score(answer_text, full_context_text) if grounded else 0.0
        if not grounded:
            grounding_level = "REFUSAL"
        elif confidence >= 0.70:
            grounding_level = "HIGH"
        else:
            grounding_level = "MEDIUM"

        msg = QAMessage(
            role="assistant",
            content=answer_text.strip(),
            citations=citations_data,
            grounded=grounded,
            grounding_confidence=confidence,
            grounding_level=grounding_level,
        )

        # Record in state QA history
        state.qa_history.append(QAMessage(role="user", content=question, citations=[]))
        state.qa_history.append(msg)
        return msg

    def process(self, state: AgentState) -> AgentState:
        """Mark agent state as ready for QA interactions."""
        state.status = AgentStatus.READY_FOR_QA
        state.add_log("Stage 7: Agent is ready for interactive grounded QA.")
        return state
