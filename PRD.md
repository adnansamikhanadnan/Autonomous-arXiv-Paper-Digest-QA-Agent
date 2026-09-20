# Product Requirements Document (PRD)

## Project: Autonomous arXiv Paper Digest & Grounded QA Agent

---

## 1. Executive Summary & Vision

### 1.1 Executive Summary
The **Autonomous arXiv Paper Digest & Grounded QA Agent** is an intelligent, stateful research companion designed to eliminate information overload for artificial intelligence and machine learning researchers, engineers, and students. The agent autonomously searches, ingests, structurally parses, and synthesizes arXiv preprints into authoritative **Executive Briefings** (featuring mandatory problem statements, technical methodologies, empirical claims, and explicit boundary limitations). Furthermore, it provides an interactive, section-cited **Grounded Question-Answering (RAG)** engine equipped with strict anti-hallucination guardrails and multi-paper comparative synthesis.

### 1.2 Vision Statement
To transform academic literature review from a tedious, hours-long manual reading process into a rapid, structured, and factually grounded interaction—enabling researchers to identify state-of-the-art breakthroughs and evaluate technical tradeoffs within seconds.

---

## 2. Problem Statement & User Personas

### 2.1 The Problem
- **Preprint Deluge**: Over 15,000 papers are submitted to arXiv monthly in computer science alone.
- **Time Inefficiency**: Researchers spend up to 70% of literature review time reading boilerplate introductions before determining whether a paper's methodology or empirical results are relevant.
- **Shallow Abstract Summaries**: Existing AI tools often summarize only the paper's abstract, missing subtle caveats, architectural nuances, dataset choices, and critical engineering limitations.
- **Hallucination Risk**: Generic LLMs hallucinate benchmark scores, author claims, and theoretical proofs when asked technical questions about newly published research.
- **Broken PDF Extraction**: Mathematical formulas, multi-column layouts, and hyphenated text frequently break standard text extractors, degrading retrieval accuracy.

### 2.2 Target User Personas

| Persona | Role / Description | Primary Needs & Pain Points |
| :--- | :--- | :--- |
| **Dr. Elena Vance** | Senior AI Research Scientist | Wants rapid distillation of novel architectures, theoretical proofs, and explicit limitations without skimming 30-page appendices. |
| **Marcus Chen** | Machine Learning Engineer | Needs exact benchmark numbers, memory footprints, speedup claims, and practical deployment constraints to evaluate SOTA models. |
| **Aarav Sharma** | Graduate Student / Reviewer | Needs to survey 5–10 candidate papers across a broad topic (e.g. *KV-cache compression*) and produce a comparative tradeoff matrix. |

---

## 3. Product Goals & Non-Goals

### 3.1 Product Goals
1. **End-to-End Autonomy**: Execute complete pipeline from natural language topic/ID to executive briefing artifact without requiring manual user intervention.
2. **Dual-Mode Query Ingestion**: Seamlessly support both broad research topic searches and exact arXiv IDs/URLs.
3. **Strict Structural Synthesis**: Enforce structured JSON and Markdown briefings with mandatory limitations, empirical claims, and discussion questions.
4. **Hierarchical Grounded Retrieval (Parent-Child RAG)**: Retrieve fine-grained micro-chunks while injecting full parent context into synthesis for zero hallucination.
5. **Anti-Hallucination & Factual Guardrails**: Enforce section/page citations and refuse out-of-scope or unsupported user questions.
6. **Multi-Paper Comparative Analysis**: Ingest and synthesize multiple candidate papers into a comparative tradeoff matrix (`--compare`).
7. **Multi-Provider & Zero-Cost Tier Accessibility**: Out-of-the-box support for Google Gemini (Free), Groq Cloud (Free Llama-3.3), Ollama (Local Open Weights), OpenAI, and a deterministic offline Mock Mode.

### 3.2 Non-Goals
1. **Full PDF Re-typesetting / Optical OCR**: Not intended as an OCR engine for scanned image bitmaps; optimized for digital text-layer PDFs.
2. **Arbitrary Web Browsing**: Agent is scoped to arXiv literature and associated metadata, not general open-web crawling.
3. **Automated Code Execution / Model Training**: Agent synthesizes and answers questions about paper text; it does not train models or run Python benchmarks directly.

---

## 4. Functional Requirements (FRs)

### FR-1: Dual-Mode Query Understanding & Normalization
- **FR-1.1**: The system must classify incoming user input into `TOPIC_SEARCH` or `SPECIFIC_PAPER`.
- **FR-1.2**: If an arXiv identifier (e.g., `2401.12345`, `2401.12345v2`, `cs/0101001`) or URL (`https://arxiv.org/abs/2401.12345`, `https://arxiv.org/pdf/2401.12345.pdf`) is detected, the agent must sanitize and extract the exact ID.
- **FR-1.3**: For natural language topics, the system must clean conversational filler, strip punctuation, and construct an optimized search term.

### FR-2: Official arXiv Atom XML Harvesting
- **FR-2.1**: The agent must query the official arXiv Atom XML API (`https://export.arxiv.org/api/query`).
- **FR-2.2**: The agent must parse title, authors, published dates, abstract, category tags, and direct PDF download links.
- **FR-2.3**: If zero results are returned, the agent must execute automatic query relaxation (dropping punctuation and stop words) to recover relevant papers.

### FR-3: Multi-Candidate Ranking & Selection
- **FR-3.1**: For topic searches with multiple returned candidates, the agent must evaluate candidate abstracts against the user's intent.
- **FR-3.2**: The agent must score semantic relevance, select the primary paper, and log the explicit selection rationale in state.

### FR-4: PDF Acquisition & Section-Aware Parsing
- **FR-4.1**: The agent must download the paper PDF, verify binary integrity, and cache it locally in `.cache_arxiv/pdfs/`.
- **FR-4.2**: The parser must identify hierarchical section boundaries (e.g., *Abstract*, *Introduction*, *Methodology*, *Experiments*, *Results*, *Limitations*, *Conclusion*) using regex layout analysis.
- **FR-4.3**: The parser must record 1-indexed page start and end numbers for each extracted section.
- **FR-4.4 (Graceful Fallback)**: If PDF download or extraction fails (due to network corruption or DRM), the agent must automatically degrade to an **Abstract Fallback Mode**, constructing structured sections from the Atom feed without crashing.

### FR-5: Hierarchical Parent-Child Indexing & RRF Search
- **FR-5.1**: Large parent chunks (~1400 characters) must be subdivided into micro-chunks (~350 characters) with sliding-window overlap.
- **FR-5.2**: The local vector index must implement **Reciprocal Rank Fusion (RRF)**, combining dense embedding cosine similarity and lexical keyword/BM25 overlap ($k=60$).
- **FR-5.3**: Retrieved search results must return child micro-chunks deduplicated by parent ID while passing the rich parent context to the synthesis prompt.

### FR-6: Executive Briefing Generation
- **FR-6.1**: The agent must generate a briefing artifact conforming to the strict Pydantic `ExecutiveBriefing` schema containing:
  1. Plain-English Summary (1 paragraph on why it matters).
  2. Problem Statement.
  3. Methodology & Technical Approach (bullet points).
  4. Key Results & Empirical Claims (quantitative benchmarks).
  5. **Explicit Limitations & Boundary Constraints** (MANDATORY).
  6. Suggested Follow-up Questions for Discussion (3+ prompts).
- **FR-6.2**: The briefing must be exportable to Markdown (`.md`) or raw JSON (`.json`).

### FR-7: Grounded Interactive Conversational QA
- **FR-7.1**: The interactive REPL must accept arbitrary follow-up user questions.
- **FR-7.2**: The agent must ground every answer in retrieved paper chunks and cite specific `[Section Name, Page X]` locations.
- **FR-7.3 (Anti-Hallucination Guardrail)**: If the retrieved excerpts lack sufficient context to answer a question, the agent must output an explicit refusal message and set `grounded = False` with `grounding_level = "REFUSAL"`.
- **FR-7.4 (Grounding Telemetry)**: The agent must compute a quantitative **Grounding Confidence Score** ($0.0 \text{ to } 1.0$) based on factual token overlap between the answer and context.

### FR-8: Multi-Paper Comparative Synthesis
- **FR-8.1**: When invoked with `--compare` or `/compare`, the agent must synthesize top candidate papers into a comparative matrix detailing core approaches, advantages, and limitations.
- **FR-8.2**: The agent must recommend an optimal reading order for the literature landscape.

---

## 5. Non-Functional Requirements (NFRs)

| Category | Requirement | Target Metric |
| :--- | :--- | :--- |
| **Performance** | Pipeline execution time (topic query to briefing) | $< 15$ seconds with cloud LLM; $< 3$ seconds in Mock mode |
| **Performance** | Local vector query & RRF ranking latency | $< 50$ milliseconds across 100 chunks |
| **Reliability** | Pipeline crash rate on malformed queries or corrupted PDFs | $0\%$ (must trigger graceful query relaxation or abstract fallback) |
| **Extensibility** | LLM provider abstraction | Plug-and-play support for Gemini, Groq, Ollama, OpenAI, Mock |
| **Portability** | Operating system support | Windows 10/11, macOS, Ubuntu/Debian Linux with zero native C-compiler builds |
| **Reproducibility** | Deterministic test suite | 100% pass rate with zero internet connection requirement |
| **Privacy / Safety** | Credential isolation | Zero hardcoded API keys; all secrets loaded from `.env` |

---

## 6. User Stories & Acceptance Criteria

### User Story 1: Topic Literature Discovery
> *As an AI researcher, I want to type "KV-cache compression for LLMs" so that the agent retrieves the best candidate papers and produces a comprehensive digest without me having to search arXiv manually.*
- **Acceptance Criteria**:
  - Agent queries arXiv Atom API and retrieves top 5 candidate papers.
  - Agent ranks papers semantically and selects the top match.
  - Agent downloads PDF, indexes sections, and displays a complete 6-part Executive Briefing.

### User Story 2: Verifying Quantitative Benchmark Claims
> *As an ML engineer, I want to ask "What speedup was achieved on 32k context lengths?" and receive the exact benchmark number with a page citation.*
- **Acceptance Criteria**:
  - Agent retrieves relevant chunks from the *Experiments/Results* section.
  - Answer quotes the quantitative speedup/memory reduction with `[Section, Page]` bracket notation.
  - Grounding confidence score exceeds $0.70$ (`HIGH`).

### User Story 3: Anti-Hallucination on Irrelevant / Trick Questions
> *As a reviewer, I want to ask an irrelevant question (e.g. "What recipe for cookies is in this paper?") and have the agent refuse rather than make up an answer.*
- **Acceptance Criteria**:
  - Agent detects insufficient context in retrieved chunks.
  - Agent outputs: *"Based on the retrieved sections of this paper, there is not enough information to answer this question."*
  - Grounded status is flagged as `False` and grounding level is `REFUSAL`.

---

## 7. Success Metrics & Key Performance Indicators (KPIs)

1. **Briefing Completeness Score**: 100% of generated briefings include all 6 required fields, specifically the mandatory *Limitations* section.
2. **Citation Precision**: $\ge 95\%$ of factual claims in QA mode include valid section names and page indices.
3. **Refusal Accuracy**: 100% refusal rate on adversarial out-of-domain questions.
4. **Offline Test Coverage**: 100% unit and integration test pass rate across all 9 test suites.
