"""Node 5: Hierarchical Parent-Child Chunking, Embeddings, and RRF Vector Index."""

import re
import math
import logging
from typing import List, Tuple, Optional, Dict, Set
from arxiv_agent.config import Config
from arxiv_agent.state import AgentState, ParsedSection, ChunkRecord, AgentStatus
from arxiv_agent.llm.provider import BaseLLMProvider, get_embedding_vector

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleVectorIndex:
    """
    High-performance local vector index implementing:
    1. Parent-Child Hierarchical retrieval
    2. Reciprocal Rank Fusion (RRF) combining dense cosine similarity and BM25 lexical ranking.
    """

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None, rrf_k: int = 60):
        self.llm = llm_provider
        self.rrf_k = rrf_k
        self.chunks: List[ChunkRecord] = []

    def clear(self) -> None:
        """Reset index."""
        self.chunks = []

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Add chunks to index and calculate embeddings if missing."""
        for chunk in chunks:
            if chunk.embedding is None:
                if self.llm:
                    chunk.embedding = self.llm.embed_text(chunk.text)
                else:
                    chunk.embedding = get_embedding_vector(chunk.text)
            self.chunks.append(chunk)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[ChunkRecord, float]]:
        """
        Query vector index using Reciprocal Rank Fusion (RRF) between Dense Semantic and Lexical rankings.
        Returns top-k deduplicated chunks with parent context.
        """
        if not self.chunks:
            return []

        # 1. Compute query vector
        if self.llm:
            query_vec = self.llm.embed_text(query_text)
        else:
            query_vec = get_embedding_vector(query_text)

        # 2. Dense Cosine Similarity Ranking
        dense_scores: List[Tuple[int, float]] = []
        for idx, chunk in enumerate(self.chunks):
            sim = cosine_similarity(query_vec, chunk.embedding or [])
            dense_scores.append((idx, sim))
        dense_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Lexical / BM25-style Keyword Overlap Ranking
        query_words: Set[str] = set(re.findall(r"\b\w{3,}\b", query_text.lower()))
        lexical_scores: List[Tuple[int, float]] = []
        for idx, chunk in enumerate(self.chunks):
            chunk_words = set(re.findall(r"\b\w{3,}\b", chunk.text.lower()))
            overlap = len(query_words.intersection(chunk_words))
            # Exact phrase bonus
            phrase_bonus = 1.0 if query_text.lower() in chunk.text.lower() else 0.0
            lexical_score = overlap + (phrase_bonus * 2.0)
            lexical_scores.append((idx, lexical_score))
        lexical_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Compute Reciprocal Rank Fusion (RRF)
        # RRF Score = 1/(k + rank_dense) + 1/(k + rank_lexical)
        dense_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_scores)}
        lexical_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(lexical_scores)}

        rrf_scores: Dict[int, float] = {}
        for idx in range(len(self.chunks)):
            r_dense = dense_rank_map[idx]
            r_lex = lexical_rank_map[idx]
            rrf = (1.0 / (self.rrf_k + r_dense)) + (1.0 / (self.rrf_k + r_lex))
            rrf_scores[idx] = rrf

        # Sort all chunks by combined RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # Deduplicate by parent_id to avoid returning redundant sub-chunks from same parent section
        results: List[Tuple[ChunkRecord, float]] = []
        seen_parents: Set[str] = set()

        for idx in sorted_indices:
            chunk = self.chunks[idx]
            parent_key = chunk.parent_id or chunk.chunk_id
            if parent_key in seen_parents and len(results) >= 2:
                continue
            seen_parents.add(parent_key)
            results.append((chunk, rrf_scores[idx]))
            if len(results) >= top_k:
                break

        return results


class VectorStoreNode:
    """
    Splits parsed sections into Hierarchical Parent-Child chunks and builds vector index.
    - Child Micro-Chunks (~350 chars): High-precision vector & keyword search.
    - Parent Chunks (~1400 chars): Full context passed to LLM for accurate synthesis.
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        parent_chunk_size: Optional[int] = None,
        parent_overlap: Optional[int] = None,
        child_chunk_size: int = 350,
        child_overlap: int = 60,
        chunk_size: Optional[int] = None,      # Backward compatibility alias
        chunk_overlap: Optional[int] = None,   # Backward compatibility alias
    ):
        self.llm = llm_provider
        self.parent_chunk_size = parent_chunk_size or chunk_size or Config.CHUNK_SIZE
        self.parent_overlap = parent_overlap or chunk_overlap or Config.CHUNK_OVERLAP
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.index = SimpleVectorIndex(llm_provider=self.llm)

    def _split_text(self, text: str, max_chars: int, overlap: int) -> List[str]:
        """Split text cleanly on paragraph or sentence boundaries."""
        if len(text) <= max_chars:
            return [text.strip()] if text.strip() else []

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(para) > max_chars:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 <= max_chars:
                            sub_chunk = f"{sub_chunk} {sent}" if sub_chunk else sent
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk.strip())
                            sub_chunk = sent
                    if sub_chunk:
                        current_chunk = sub_chunk
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_sections(self, sections: List[ParsedSection]) -> List[ChunkRecord]:
        """
        Build Parent-Child Hierarchical chunks from parsed sections.
        """
        all_child_chunks: List[ChunkRecord] = []

        for sec_idx, section in enumerate(sections, 1):
            # 1. Create Parent Chunks
            parent_texts = self._split_text(
                section.content, self.parent_chunk_size, self.parent_overlap
            )

            for p_idx, parent_text in enumerate(parent_texts, 1):
                parent_id = f"sec_{sec_idx}_p{p_idx}"

                # Calculate page estimate
                page_num = section.page_start
                if section.page_end > section.page_start and len(parent_texts) > 1:
                    page_fraction = (p_idx - 1) / max(1, len(parent_texts) - 1)
                    page_num = int(
                        section.page_start
                        + page_fraction * (section.page_end - section.page_start)
                    )

                # 2. Subdivide into Child Micro-Chunks
                child_texts = self._split_text(
                    parent_text, self.child_chunk_size, self.child_overlap
                )

                for c_idx, child_text in enumerate(child_texts, 1):
                    child_id = f"{parent_id}_c{c_idx}"
                    all_child_chunks.append(
                        ChunkRecord(
                            chunk_id=child_id,
                            parent_id=parent_id,
                            section_title=section.heading,
                            page_number=page_num,
                            text=child_text,
                            parent_text=parent_text,
                            token_count=len(child_text.split()),
                        )
                    )

        return all_child_chunks

    def process(self, state: AgentState) -> AgentState:
        """Execute node step on shared agent state."""
        state.status = AgentStatus.INDEXING_CHUNKS
        state.add_log("Stage 5: Building Parent-Child hierarchical index and computing RRF embeddings...")

        if not state.parsed_sections:
            state.add_error("No parsed sections found to chunk.")
            state.status = AgentStatus.FAILED
            return state

        # Create hierarchical chunks
        chunks = self.chunk_sections(state.parsed_sections)
        state.chunks = chunks

        # Index in vector store
        self.index.clear()
        self.index.add_chunks(chunks)

        state.add_log(
            f"Successfully built Parent-Child index with {len(chunks)} micro-chunks."
        )
        return state
