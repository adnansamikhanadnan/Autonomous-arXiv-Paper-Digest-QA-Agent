"""Tests for Node 4b: PDF Section Parsing & Degradation Fallback."""

# pyrefly: ignore [missing-import]
import pytest
from arxiv_agent.state import AgentState, PaperMetadata, ParsedSection
from arxiv_agent.nodes.pdf_parser import PDFParserNode, SECTION_HEADER_REGEX, clean_pdf_text


def test_section_header_regex():
    assert SECTION_HEADER_REGEX.match("1 Introduction")
    assert SECTION_HEADER_REGEX.match("2 Related Work")
    assert SECTION_HEADER_REGEX.match("3 Methodology")
    assert SECTION_HEADER_REGEX.match("3.1 Model Architecture")
    assert SECTION_HEADER_REGEX.match("4 Experiments and Results")
    assert SECTION_HEADER_REGEX.match("5 Discussion")
    assert SECTION_HEADER_REGEX.match("6 Limitations and Broader Impact")
    assert SECTION_HEADER_REGEX.match("7 Conclusion")
    assert SECTION_HEADER_REGEX.match("References")
    assert SECTION_HEADER_REGEX.match("Appendix A")


def test_clean_pdf_text():
    raw = "This is a transfor-\nmer architecture with  extra   spaces\nand standard flow."
    cleaned = clean_pdf_text(raw)
    assert "transformer architecture" in cleaned
    assert "  " not in cleaned


def test_fallback_from_metadata():
    node = PDFParserNode()
    state = AgentState(
        user_query="2401.12345",
        selected_paper=PaperMetadata(
            arxiv_id="2401.12345",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            abstract="We propose the Transformer, a model architecture relying entirely on an attention mechanism...",
            published="2017-06-12",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            entry_url="https://arxiv.org/abs/1706.03762",
            categories=["cs.CL", "cs.AI"],
        ),
        pdf_path=None,  # Missing PDF -> fallback mode
    )

    out_state = node.process(state)
    assert out_state.is_fallback_mode is True
    assert len(out_state.parsed_sections) >= 2
    assert "Attention Is All You Need" in out_state.full_text
    assert "Transformer" in out_state.full_text
