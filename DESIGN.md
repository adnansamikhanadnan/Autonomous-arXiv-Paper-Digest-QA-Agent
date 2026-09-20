# Technical & Component Design Document

## Project: Autonomous arXiv Paper Digest & Grounded QA Agent

---

## 1. Design Principles & Goals

1. **State Immutability & Traceability**: Maintain a centralized, typed Pydantic state container passed across nodes. Every node reads from state and returns an updated state object with audit logs.
2. **Deterministic Fallbacks**: Every external dependency (network, PDF binary, LLM provider) must have a deterministic, graceful fallback mechanism.
3. **High-Precision Retrieval**: Optimize RAG with **Parent-Child Hierarchical Chunking** and **Reciprocal Rank Fusion (RRF)** to eliminate context fragmentation.
4. **Factual Grounding by Construction**: Enforce `[Section, Page]` bracket citations and verify token overlap mathematically before emitting answers.

---

## 2. Core Data Contracts & Schemas

All data contracts are implemented in `arxiv_agent/state.py` using **Pydantic v2**:

### 2.1 `PaperMetadata`
Represents an arXiv preprint retrieved from Atom XML:
```python
class PaperMetadata(BaseModel):
    arxiv_id: str                      # e.g., '2401.12345'
    title: str                         # Full paper title
    authors: List[str]                 # List of authors
    abstract: str                      # Complete abstract text
    published: str                     # Submission date (YYYY-MM-DD)
    pdf_url: str                       # Direct PDF link
    entry_url: str                     # arXiv landing page URL
    categories: List[str]              # e.g., ['cs.CL', 'cs.AI']
    comment: Optional[str] = None      # Conference notes, page count
    relevance_score: Optional[float]   # Semantic topic match score (0.0 to 1.0)
    selection_rationale: Optional[str] # LLM rationale for selection
```

### 2.2 `ParsedSection`
Represents a structured section extracted from the PDF:
```python
class ParsedSection(BaseModel):
    heading: str                       # e.g., 'Introduction', 'Methodology'
    content: str                       # Extracted clean section body
    page_start: int                    # 1-indexed start page
    page_end: int                      # 1-indexed end page
```

### 2.3 `ChunkRecord`
Represents hierarchical parent-child chunks for vector search:
```python
class ChunkRecord(BaseModel):
    chunk_id: str                      # e.g., 'sec_3_p1_c2'
    parent_id: Optional[str]           # e.g., 'sec_3_p1'
    section_title: str                 # Extracted heading
    page_number: int                   # Page number estimate
    text: str                          # Child micro-chunk (~350 chars) for search
    parent_text: Optional[str]         # Parent macro-chunk (~1400 chars) for LLM context
    token_count: Optional[int]         # Approximate token count
    embedding: Optional[List[float]]   # Dense vector embedding
```

### 2.4 `ExecutiveBriefing`
The primary structured digest artifact required by researchers:
```python
class ExecutiveBriefing(BaseModel):
    title: str
    authors: List[str]
    arxiv_id: str
    publish_date: str
    link: str
    plain_english_summary: str         # 1-paragraph summary (why it matters)
    problem_statement: str             # Core challenge addressed
    method_approach: List[str]         # Key architectural/methodology bullet points
    key_results_claims: List[str]      # Quantitative findings & benchmark numbers
    limitations: List[str]             # MANDATORY explicit caveats & boundary constraints
    suggested_follow_up_questions: List[str] # 3+ probing discussion questions
```

### 2.5 `QAMessage`
A turn in the conversational QA loop with quantitative grounding telemetry:
```python
class QAMessage(BaseModel):
    role: str                          # 'user' or 'assistant'
    content: str                       # Message text
    citations: List[Dict[str, Any]]    # Section, page, score, preview
    grounded: bool                     # True if factually grounded; False on refusal
    grounding_confidence: float        # Score from 0.0 to 1.0
    grounding_level: str               # 'HIGH', 'MEDIUM', or 'REFUSAL'
```

---

## 3. Mathematical Formulations & Algorithms

### 3.1 Cosine Similarity
For two embedding vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:
$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

### 3.2 Reciprocal Rank Fusion (RRF)
Combines dense embedding retrieval and lexical keyword overlap to produce a robust hybrid rank:
$$\text{RRF Score}(d) = \frac{1}{k + \text{Rank}_{\text{dense}}(d)} + \frac{1}{k + \text{Rank}_{\text{lexical}}(d)}$$
Where $k = 60$ is the standard smoothing parameter preventing high ranks from dominating.

- **Lexical Score Formulation**:
  $$\text{Score}_{\text{lexical}}(d) = |W_q \cap W_d| + 2.0 \cdot \mathbb{I}(\text{Phrase}_q \subseteq \text{Text}_d)$$
  Where $W_q, W_d$ are the substantive word token sets of query $q$ and document $d$.

### 3.3 Quantitative Grounding Confidence Score
Evaluates the factual fidelity of an assistant answer $A$ against the retrieved context $C$:
$$\text{OverlapRatio}(A, C) = \frac{|W_A^{\text{substantive}} \cap W_C^{\text{substantive}}|}{|W_A^{\text{substantive}}|}$$

$$\text{Confidence}(A, C) = \begin{cases} 
0.0 & \text{if } A \text{ is a refusal} \\
\min\left(1.0, \max\left(0.0, 0.4 + 0.6 \cdot \text{OverlapRatio}(A, C)\right)\right) & \text{otherwise}
\end{cases}$$

- **Grounding Tier Assignment**:
  - $\text{Confidence} \ge 0.70 \implies \mathbf{HIGH}$
  - $0.0 < \text{Confidence} < 0.70 \implies \mathbf{MEDIUM}$
  - $\text{Confidence} = 0.0 \implies \mathbf{REFUSAL}$

---

## 4. Component Deep Dive

### 4.1 Regular Expression Engineering
The system utilizes robust regex patterns for entity extraction and layout parsing:

- **arXiv ID Identification**:
  ```python
  ARXIV_ID_PATTERN = re.compile(
      r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z\-]+(?:\.[a-z]{2})?/[0-9]{7})',
      re.IGNORECASE
  )
  ```
- **Academic Header Boundaries**:
  ```python
  SECTION_HEADER_PATTERN = re.compile(
      r'^(?:\d+\.?\s+)?(Abstract|Introduction|Background|Related Work|Methodology|'
      r'Proposed Method|Architecture|Experiments?|Results|Discussion|'
      r'Limitations?|Conclusion|References?)\b',
      re.IGNORECASE | re.MULTILINE
  )
  ```

### 4.2 Parent-Child Chunking Strategy

```
Raw Section Content (~5,000 chars)
  │
  ├──► Parent Chunk 1 (~1,400 chars, 200 overlap)
  │     ├──► Child Chunk 1.1 (~350 chars, 60 overlap) ──► Indexed with Embedding
  │     ├──► Child Chunk 1.2 (~350 chars, 60 overlap) ──► Indexed with Embedding
  │     └──► Child Chunk 1.3 (~350 chars, 60 overlap) ──► Indexed with Embedding
  │
  └──► Parent Chunk 2 (~1,400 chars, 200 overlap)
        ├──► Child Chunk 2.1 (~350 chars, 60 overlap) ──► Indexed with Embedding
        └──► Child Chunk 2.2 (~350 chars, 60 overlap) ──► Indexed with Embedding
```

**Retrieval Flow**:
1. Child chunks are queried using RRF.
2. The search results are deduplicated by `parent_id`.
3. The LLM synthesis prompt is populated with the complete `parent_text`, ensuring the LLM understands surrounding equations, context, and definitions.

---

## 5. User Interface & CLI Specification

The CLI (`arxiv_agent/cli.py`) supports full non-interactive scripting and rich interactive exploration:

### CLI Options Matrix
| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | Positional | `None` | Research topic query or arXiv ID/URL |
| `--provider` | `-p` | `.env` | LLM provider: `gemini`, `groq`, `openai`, `ollama`, `mock` |
| `--model` | `-m` | Provider default | Model name override |
| `--output` | `-o` | `None` | Save briefing to file (`.md` or `.json`) |
| `--json` | - | `False` | Stream raw JSON briefing to stdout |
| `--compare` | - | `False` | Generate multi-paper comparative synthesis matrix |
| `--mock` / `--offline` | - | `False` | Force offline deterministic mock provider |
| `--interactive` | `-i` | `False` | Launch interactive conversational QA loop |

### Interactive REPL Commands
- `/briefing`: Redisplay the formatted Executive Briefing.
- `/compare`: Trigger multi-candidate comparative synthesis table.
- `/stats`: Display pipeline telemetry (section counts, tokens, chunk counts, fallback status).
- `/chunks`: Inspect top indexed chunks with character offsets and page numbers.
- `/export`: Export complete session (briefing + comparative matrix + QA history) to `qa_export.md`.
- `exit` / `quit`: Gracefully terminate session.
