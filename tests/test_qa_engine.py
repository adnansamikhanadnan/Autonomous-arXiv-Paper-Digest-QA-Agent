"""Tests for Node 7: Grounded QA Engine, Citations & Anti-Hallucination Guardrails."""

import pytest
from arxiv_agent.state import AgentState, PaperMetadata, ChunkRecord
from arxiv_agent.nodes.vector_store import VectorStoreNode
from arxiv_agent.nodes.qa_engine import QAEngineNode
from arxiv_agent.llm.provider import MockProvider


def test_qa_grounded_answer():
    mock_llm = MockProvider()
    vec_node = VectorStoreNode(llm_provider=mock_llm)
    
    # Pre-populate index
    chunks = [
        ChunkRecord(
            chunk_id="c1",
            section_title="Methodology",
            page_number=3,
            text="The proposed method uses attention-head importance scoring to prune the KV cache.",
        ),
        ChunkRecord(
            chunk_id="c2",
            section_title="Results",
            page_number=5,
            text="Table 2 reports a 65% memory reduction with less than 0.8% perplexity degradation.",
        ),
    ]
    vec_node.index.add_chunks(chunks)

    qa_node = QAEngineNode(vector_node=vec_node, llm_provider=mock_llm)

    state = AgentState(
        user_query="2401.12345",
        selected_paper=PaperMetadata(
            arxiv_id="2401.12345",
            title="KV-Cache Compression Paper",
            authors=["Alice"],
            abstract="Summary",
            published="2024-01-15",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
            entry_url="https://arxiv.org/abs/2401.12345",
        ),
    )

    msg = qa_node.answer_question("How much memory does the method save?", state)
    assert msg.role == "assistant"
    assert len(msg.citations) > 0
    assert "Section 3" in msg.content or "Methodology" in msg.content or "65%" in msg.content
    assert msg.grounded is True
    assert len(state.qa_history) == 2  # user + assistant turns


def test_qa_anti_hallucination_refusal():
    mock_llm = MockProvider()
    vec_node = VectorStoreNode(llm_provider=mock_llm)
    qa_node = QAEngineNode(vector_node=vec_node, llm_provider=mock_llm)

    state = AgentState(
        user_query="2401.12345",
        selected_paper=PaperMetadata(
            arxiv_id="2401.12345",
            title="KV-Cache Compression Paper",
            authors=["Alice"],
            abstract="Summary",
            published="2024-01-15",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
            entry_url="https://arxiv.org/abs/2401.12345",
        ),
    )

    # Ask unsupported question
    msg = qa_node.answer_question("What is the recipe for chocolate cake mentioned in this paper?", state)
    assert msg.grounded is False or "does not contain information" in msg.content.lower()
