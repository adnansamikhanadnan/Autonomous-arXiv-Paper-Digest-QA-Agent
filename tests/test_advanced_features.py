"""Tests for Advanced Features: Parent-Child Chunking, RRF Fusion, Grounding Score, Comparative Synthesis."""

# pyrefly: ignore [missing-import]
import pytest
from arxiv_agent.state import ParsedSection, ChunkRecord, PaperMetadata
from arxiv_agent.nodes.vector_store import VectorStoreNode, SimpleVectorIndex
from arxiv_agent.nodes.qa_engine import calculate_grounding_score
from arxiv_agent.nodes.comparative_synthesizer import ComparativeSynthesizerNode
from arxiv_agent.llm.provider import MockProvider


def test_parent_child_chunking():
    node = VectorStoreNode(parent_chunk_size=500, child_chunk_size=150, child_overlap=30)
    sections = [
        ParsedSection(
            heading="Methodology",
            content="We propose a dynamic KV-cache eviction algorithm. " * 20,
            page_start=3,
            page_end=4,
        )
    ]

    chunks = node.chunk_sections(sections)
    assert len(chunks) > 1
    # Check parent_id and parent_text links
    for chunk in chunks:
        assert chunk.parent_id is not None
        assert chunk.parent_text is not None
        assert len(chunk.parent_text) >= len(chunk.text)
        assert chunk.text in chunk.parent_text


def test_rrf_vector_search():
    index = SimpleVectorIndex(rrf_k=60)
    chunks = [
        ChunkRecord(
            chunk_id="p1_c1",
            parent_id="p1",
            section_title="Methodology",
            page_number=3,
            text="Dynamic KV-cache eviction pruning policy with sink token preservation.",
            parent_text="Full Section Methodology on Dynamic KV-cache eviction...",
        ),
        ChunkRecord(
            chunk_id="p1_c2",
            parent_id="p1",
            section_title="Methodology",
            page_number=3,
            text="Attention-head importance thresholding determines memory eviction.",
            parent_text="Full Section Methodology on Dynamic KV-cache eviction...",
        ),
        ChunkRecord(
            chunk_id="p2_c1",
            parent_id="p2",
            section_title="Experiments",
            page_number=5,
            text="Evaluation on A100 GPU cluster across 32k context benchmarks.",
            parent_text="Full Experiments Section text...",
        ),
    ]
    index.add_chunks(chunks)

    results = index.query("How does dynamic KV-cache eviction work?", top_k=2)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert "Dynamic KV-cache" in top_chunk.text or "eviction" in top_chunk.text
    assert score > 0.0


def test_calculate_grounding_score():
    context = (
        "The proposed approach dynamically scores token importance and evicts tokens below the threshold, "
        "retaining sink tokens to preserve generation stability. Table 2 reports a 65% memory reduction."
    )

    grounded_answer = "Based on Section 3, the method uses token importance scoring to evict tokens, achieving a 65% memory reduction."
    score_high = calculate_grounding_score(grounded_answer, context)
    assert score_high >= 0.70

    refusal_answer = "Based on the retrieved sections of this paper, there is not enough information to answer this question."
    score_refusal = calculate_grounding_score(refusal_answer, context)
    assert score_refusal == 0.0


def test_comparative_synthesizer():
    mock_llm = MockProvider()
    synthesizer = ComparativeSynthesizerNode(llm_provider=mock_llm)

    candidates = [
        PaperMetadata(
            arxiv_id="2401.12345",
            title="KV-Cache Compression in LLMs",
            authors=["Alice"],
            abstract="Dynamic cache eviction for fast decoding.",
            published="2024-01-15",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
            entry_url="https://arxiv.org/abs/2401.12345",
            categories=["cs.CL"],
        ),
        PaperMetadata(
            arxiv_id="2305.99999",
            title="Quantized Attention Caching",
            authors=["Bob"],
            abstract="4-bit quantization of keys and values.",
            published="2023-05-10",
            pdf_url="https://arxiv.org/pdf/2305.99999.pdf",
            entry_url="https://arxiv.org/abs/2305.99999",
            categories=["cs.AI"],
        ),
    ]

    analysis = synthesizer.synthesize("KV-cache compression", candidates)
    assert analysis.topic == "KV-cache compression"
    assert len(analysis.papers) >= 2
    assert len(analysis.synthesis_summary) > 0
    assert len(analysis.recommended_reading_order) > 0
