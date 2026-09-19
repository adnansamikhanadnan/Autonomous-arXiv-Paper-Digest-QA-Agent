"""Tests for Failure Cases & Graceful Degradation."""

import pytest
from arxiv_agent.state import AgentStatus
from arxiv_agent.graph import ArxivAgentGraph
from arxiv_agent.llm.provider import MockProvider
from arxiv_agent.nodes.arxiv_retrieval import PaperMetadata


def test_failure_case_zero_candidate_papers(monkeypatch):
    """Failure Scenario 1: arXiv API returns 0 candidate papers for a bizarre/vague query."""
    mock_llm = MockProvider()
    graph = ArxivAgentGraph(llm_provider=mock_llm)

    # Mock empty search results
    monkeypatch.setattr(graph.retrieval_node, "search_by_topic", lambda topic, max_results=5: [])

    state = graph.run("xyznonexistenttopic123456789")

    assert state.status == AgentStatus.FAILED
    assert len(state.errors) > 0
    assert "No papers found on arXiv" in state.errors[0]
    assert state.executive_briefing is None


def test_failure_case_pdf_download_failure(monkeypatch):
    """Failure Scenario 2: PDF download fails (404/network error). Fallback to abstract mode."""
    mock_llm = MockProvider()
    graph = ArxivAgentGraph(llm_provider=mock_llm)

    # Mock paper metadata return
    monkeypatch.setattr(
        graph.retrieval_node,
        "fetch_by_id",
        lambda aid: [
            PaperMetadata(
                arxiv_id="2401.99999",
                title="Paper with Broken PDF Link",
                authors=["Test Author"],
                abstract="This is a test abstract that serves as fallback content.",
                published="2024-01-01",
                pdf_url="https://arxiv.org/pdf/invalid_url.pdf",
                entry_url="https://arxiv.org/abs/2401.99999",
                categories=["cs.AI"],
            )
        ],
    )

    # Force PDF fetcher to fail and return None
    monkeypatch.setattr(graph.fetcher_node, "fetch_pdf", lambda aid, url: None)

    state = graph.run("2401.99999")

    # Pipeline must NOT crash and should successfully generate briefing in fallback mode
    assert state.status == AgentStatus.READY_FOR_QA
    assert state.is_fallback_mode is True
    assert state.executive_briefing is not None
    assert len(state.warnings) > 0
    assert any("abstract/metadata mode" in w for w in state.warnings)
    assert len(state.chunks) > 0


def test_failure_case_corrupted_pdf_file(monkeypatch, tmp_path):
    """Failure Scenario 3: Corrupted or unparseable PDF file triggers fallback gracefully."""
    mock_llm = MockProvider()
    graph = ArxivAgentGraph(llm_provider=mock_llm)

    # Create dummy empty file pretending to be a PDF
    dummy_file = tmp_path / "corrupt.pdf"
    dummy_file.write_bytes(b"%PDF-1.4\nCorrupted binary garbage without valid text stream\n%%EOF")

    monkeypatch.setattr(
        graph.retrieval_node,
        "fetch_by_id",
        lambda aid: [
            PaperMetadata(
                arxiv_id="2401.88888",
                title="Paper with Corrupt PDF",
                authors=["Corrupt Author"],
                abstract="Fallback abstract for corrupt PDF test.",
                published="2024-01-01",
                pdf_url="https://arxiv.org/pdf/corrupt.pdf",
                entry_url="https://arxiv.org/abs/2401.88888",
                categories=["cs.AI"],
            )
        ],
    )
    monkeypatch.setattr(graph.fetcher_node, "fetch_pdf", lambda aid, url: str(dummy_file))

    state = graph.run("2401.88888")

    assert state.status == AgentStatus.READY_FOR_QA
    assert state.is_fallback_mode is True
    assert state.executive_briefing is not None
