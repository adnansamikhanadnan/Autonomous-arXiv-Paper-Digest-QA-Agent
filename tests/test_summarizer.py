"""Tests for Node 6: Executive Briefing Generation & Schema Enforcement."""

import pytest
from arxiv_agent.state import AgentState, PaperMetadata, ParsedSection, ExecutiveBriefing
from arxiv_agent.nodes.summarizer import SummarizerNode, format_briefing_markdown
from arxiv_agent.llm.provider import MockProvider


def test_executive_briefing_schema():
    briefing = ExecutiveBriefing(
        title="Test Research Paper",
        authors=["Alice", "Bob"],
        arxiv_id="2401.12345",
        publish_date="2024-01-15",
        link="https://arxiv.org/abs/2401.12345",
        plain_english_summary="This paper matters because it speeds up AI inference by 3x.",
        problem_statement="High latency during inference.",
        method_approach=["Token pruning", "Memory optimization"],
        key_results_claims=["3x speedup", "50% memory reduction"],
        limitations=["Limited to dense models", "Requires warmup"],
        suggested_follow_up_questions=["Does it work with MoE models?"],
    )

    assert briefing.title == "Test Research Paper"
    assert len(briefing.limitations) == 2
    assert len(briefing.method_approach) == 2


def test_summarizer_node_with_mock_llm():
    mock_llm = MockProvider()
    node = SummarizerNode(llm_provider=mock_llm)

    state = AgentState(
        user_query="2401.12345",
        selected_paper=PaperMetadata(
            arxiv_id="2401.12345",
            title="KV-Cache Compression and Acceleration in Large Language Models",
            authors=["Alice Chen", "Bob Smith"],
            abstract="We present a novel dynamic KV-cache eviction policy...",
            published="2024-01-15",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
            entry_url="https://arxiv.org/abs/2401.12345",
            categories=["cs.CL"],
        ),
        parsed_sections=[
            ParsedSection(
                heading="Introduction",
                content="LLMs are memory bottlenecked...",
                page_start=1,
                page_end=1,
            )
        ],
    )

    out_state = node.process(state)
    assert out_state.executive_briefing is not None
    b = out_state.executive_briefing
    assert len(b.limitations) > 0
    assert len(b.method_approach) > 0
    assert len(b.key_results_claims) > 0
    assert len(b.suggested_follow_up_questions) > 0


def test_format_briefing_markdown():
    briefing = ExecutiveBriefing(
        title="Test Title",
        authors=["Alice", "Bob"],
        arxiv_id="2401.12345",
        publish_date="2024-01-15",
        link="https://arxiv.org/abs/2401.12345",
        plain_english_summary="Summary text.",
        problem_statement="Problem text.",
        method_approach=["Method 1", "Method 2"],
        key_results_claims=["Result 1"],
        limitations=["Limitation 1"],
        suggested_follow_up_questions=["Question 1?"],
    )

    md = format_briefing_markdown(briefing)
    assert "# Executive Briefing: Test Title" in md
    assert "## 📌 Plain-English Summary" in md
    assert "## ⚠️ Explicit Limitations" in md
    assert "Limitation 1" in md
