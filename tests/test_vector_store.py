"""Tests for Node 5: Section-Aware Chunking, Embeddings, and Local Vector Index."""

import pytest
from arxiv_agent.state import AgentState, ParsedSection, ChunkRecord
from arxiv_agent.nodes.vector_store import VectorStoreNode, SimpleVectorIndex, cosine_similarity
from arxiv_agent.llm.provider import get_embedding_vector


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3), 0.001) == 0.0


def test_embedding_determinism():
    text = "KV-cache compression enables long-context LLM inference."
    emb1 = get_embedding_vector(text)
    emb2 = get_embedding_vector(text)

    assert len(emb1) == 128
    assert emb1 == emb2
    assert pytest.approx(cosine_similarity(emb1, emb2), 0.0001) == 1.0


def test_chunking_sections():
    node = VectorStoreNode(chunk_size=100, chunk_overlap=20)
    sections = [
        ParsedSection(
            heading="Introduction",
            content="Large language models require extensive GPU memory for KV cache. " * 3,
            page_start=1,
            page_end=2,
        ),
        ParsedSection(
            heading="Methodology",
            content="We introduce a dynamic pruning policy that discards redundant tokens. " * 3,
            page_start=3,
            page_end=4,
        ),
    ]

    chunks = node.chunk_sections(sections)
    assert len(chunks) >= 2
    assert all(isinstance(c, ChunkRecord) for c in chunks)
    assert any("Introduction" in c.section_title for c in chunks)
    assert any("Methodology" in c.section_title for c in chunks)


def test_vector_index_query():
    index = SimpleVectorIndex()
    chunks = [
        ChunkRecord(
            chunk_id="c1",
            section_title="Methodology",
            page_number=3,
            text="The dynamic KV-cache eviction algorithm uses attention sinks and importance scoring.",
        ),
        ChunkRecord(
            chunk_id="c2",
            section_title="Experiments",
            page_number=5,
            text="We benchmarked on A100 GPUs and measured throughput across 32k token sequence lengths.",
        ),
        ChunkRecord(
            chunk_id="c3",
            section_title="Related Work",
            page_number=2,
            text="Prior work focused on 4-bit and 8-bit quantization for weights rather than activation caches.",
        ),
    ]
    index.add_chunks(chunks)

    # Query matching c1
    results = index.query("How does the dynamic eviction algorithm work?", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert top_chunk.section_title == "Methodology"
