"""Tests for End-to-End Stateful Agent Graph Execution."""

import pytest
from arxiv_agent.state import AgentStatus, IntentType
from arxiv_agent.graph import ArxivAgentGraph
from arxiv_agent.llm.provider import MockProvider
from arxiv_agent.nodes.arxiv_retrieval import PaperMetadata


def test_end_to_end_graph_specific_paper(monkeypatch):
    mock_llm = MockProvider()
    graph = ArxivAgentGraph(llm_provider=mock_llm)

    # Mock arXiv retrieval to avoid live network query during unit tests
    def mock_fetch_by_id(arxiv_id):
        return [
            PaperMetadata(
                arxiv_id="2401.12345",
                title="KV-Cache Compression and Acceleration in Large Language Models",
                authors=["Alice Chen", "Bob Smith"],
                abstract="We propose dynamic KV-cache pruning...",
                published="2024-01-15",
                pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
                entry_url="https://arxiv.org/abs/2401.12345",
                categories=["cs.CL", "cs.AI"],
            )
        ]

    monkeypatch.setattr(graph.retrieval_node, "fetch_by_id", mock_fetch_by_id)

    # Run state graph
    state = graph.run("2401.12345")

    # Verify state transitions
    assert state.status == AgentStatus.READY_FOR_QA
    assert state.intent == IntentType.SPECIFIC_PAPER
    assert state.selected_paper is not None
    assert state.selected_paper.arxiv_id == "2401.12345"
    assert state.executive_briefing is not None
    assert len(state.chunks) > 0

    # Verify QA Loop execution
    qa_msg = graph.ask("What are the key results?")
    assert qa_msg.role == "assistant"
    assert len(qa_msg.content) > 0
    assert len(state.qa_history) == 2


def test_end_to_end_graph_topic_search(monkeypatch):
    mock_llm = MockProvider()
    graph = ArxivAgentGraph(llm_provider=mock_llm)

    # Mock search by topic with multiple candidates
    def mock_search_by_topic(topic, max_results=5):
        return [
            PaperMetadata(
                arxiv_id="2401.12345",
                title="KV-Cache Compression and Acceleration in Large Language Models",
                authors=["Alice Chen"],
                abstract="We propose dynamic KV-cache pruning...",
                published="2024-01-15",
                pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
                entry_url="https://arxiv.org/abs/2401.12345",
                categories=["cs.CL"],
            ),
            PaperMetadata(
                arxiv_id="2305.99999",
                title="A Survey on Model Quantization",
                authors=["Charlie"],
                abstract="A broad survey on quantization...",
                published="2023-05-10",
                pdf_url="https://arxiv.org/pdf/2305.99999.pdf",
                entry_url="https://arxiv.org/abs/2305.99999",
                categories=["cs.LG"],
            ),
        ]

    monkeypatch.setattr(graph.retrieval_node, "search_by_topic", mock_search_by_topic)

    state = graph.run("recent work on KV-cache compression for LLMs")

    assert state.status == AgentStatus.READY_FOR_QA
    assert state.intent == IntentType.TOPIC_SEARCH
    assert len(state.candidate_papers) == 2
    assert state.selected_paper is not None
    assert state.executive_briefing is not None
