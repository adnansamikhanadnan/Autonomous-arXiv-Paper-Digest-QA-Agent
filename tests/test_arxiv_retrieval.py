"""Tests for Node 2: arXiv API Retrieval & XML Parsing."""

import xml.etree.ElementTree as ET
from arxiv_agent.state import AgentState, IntentType
from arxiv_agent.nodes.arxiv_retrieval import ArxivRetrievalNode, clean_text

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2024-01-15T12:00:00Z</published>
    <title>KV-Cache Compression and Acceleration in Large Language Models</title>
    <summary>This paper proposes a novel dynamic eviction strategy for KV-cache...</summary>
    <author><name>Alice Chen</name></author>
    <author><name>Bob Smith</name></author>
    <arxiv:comment>Accepted at ICLR 2024</arxiv:comment>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <link href="https://arxiv.org/pdf/2401.12345.pdf" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>
"""


def test_parse_entry():
    node = ArxivRetrievalNode()
    root = ET.fromstring(SAMPLE_ATOM_XML)
    entry_elem = root.find("{http://www.w3.org/2005/Atom}entry")
    assert entry_elem is not None
    
    paper = node._parse_entry(entry_elem)
    assert paper is not None
    assert paper.arxiv_id == "2401.12345"
    assert "KV-Cache Compression" in paper.title
    assert paper.authors == ["Alice Chen", "Bob Smith"]
    assert paper.published == "2024-01-15"
    assert "cs.CL" in paper.categories
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345.pdf"
    assert paper.comment == "Accepted at ICLR 2024"


def test_clean_text():
    raw = "  Line 1 with   extra spaces \n and newlines.  \t "
    assert clean_text(raw) == "Line 1 with extra spaces and newlines."


def test_retrieval_node_process(monkeypatch):
    node = ArxivRetrievalNode()

    # Mock _execute_query to return our parsed paper
    root = ET.fromstring(SAMPLE_ATOM_XML)
    entry_elem = root.find("{http://www.w3.org/2005/Atom}entry")
    mock_paper = node._parse_entry(entry_elem)

    monkeypatch.setattr(node, "_execute_query", lambda params: [mock_paper])

    state = AgentState(
        user_query="2401.12345",
        intent=IntentType.SPECIFIC_PAPER,
        extracted_arxiv_id="2401.12345"
    )

    out_state = node.process(state)
    assert len(out_state.candidate_papers) == 1
    assert out_state.candidate_papers[0].arxiv_id == "2401.12345"
