"""Tests for Node 1: Query Understanding & Intent Parsing."""

import pytest
from arxiv_agent.state import AgentState, IntentType
from arxiv_agent.nodes.query_understanding import QueryUnderstandingNode


def test_extract_modern_arxiv_id():
    node = QueryUnderstandingNode()
    
    # Plain ID
    assert node.extract_arxiv_id("2401.12345") == "2401.12345"
    assert node.extract_arxiv_id("2305.18290v2") == "2305.18290v2"
    
    # Text with ID
    assert node.extract_arxiv_id("Can you summarize 2401.12345 for me?") == "2401.12345"
    assert node.extract_arxiv_id("arXiv:2401.12345") == "2401.12345"


def test_extract_arxiv_urls():
    node = QueryUnderstandingNode()
    
    assert node.extract_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"
    assert node.extract_arxiv_id("https://arxiv.org/pdf/2401.12345.pdf") == "2401.12345"
    assert node.extract_arxiv_id("http://export.arxiv.org/abs/2301.07069v1") == "2301.07069v1"


def test_extract_old_style_arxiv_id():
    node = QueryUnderstandingNode()
    
    assert node.extract_arxiv_id("cs/0601001") == "cs/0601001"
    assert node.extract_arxiv_id("https://arxiv.org/abs/math.GT/0309136") == "math.GT/0309136"
    assert node.extract_arxiv_id("hep-th/9912012") == "hep-th/9912012"


def test_topic_query_cleaning():
    node = QueryUnderstandingNode()
    
    assert node.clean_topic_query("recent work on KV-cache compression for LLMs") == "KV-cache compression for LLMs"
    assert node.clean_topic_query("can you please find papers on diffusion models?") == "diffusion models"
    assert node.clean_topic_query("Tell me about Mamba state space models.") == "Mamba state space models"


def test_process_specific_paper_state():
    node = QueryUnderstandingNode()
    state = AgentState(user_query="https://arxiv.org/abs/2401.12345")
    
    out_state = node.process(state)
    assert out_state.intent == IntentType.SPECIFIC_PAPER
    assert out_state.extracted_arxiv_id == "2401.12345"
    assert out_state.cleaned_query == "2401.12345"


def test_process_topic_search_state():
    node = QueryUnderstandingNode()
    state = AgentState(user_query="recent work on KV-cache compression for LLMs")
    
    out_state = node.process(state)
    assert out_state.intent == IntentType.TOPIC_SEARCH
    assert out_state.extracted_arxiv_id is None
    assert "KV-cache compression" in out_state.cleaned_query
