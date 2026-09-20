# Operational Memory & System Context

## Project: Autonomous arXiv Paper Digest & Grounded QA Agent

---

## 1. Project Context & Identity

- **Repository**: Autonomous arXiv Paper Digest & Grounded QA Agent
- **Corpus**: `adnansamikhanadnan/Autonomous-arXiv-Paper-Digest-QA-Agent`
- **Core Purpose**: Autonomous discovery, hierarchical PDF parsing, structured executive summarization (with mandatory limitations), and grounded conversational QA (RAG) over arXiv research papers.
- **Language & Runtime**: Python 3.10+ (tested on Python 3.13 / Windows 11 / Linux / macOS).
- **Primary Dependencies**: `pydantic` (v2), `pymupdf` (FitZ), `python-dotenv`, `urllib` (standard library), `pytest`.

---

## 2. Directory Tree & File Inventory

```
Project/
├── .cache_arxiv/                 # Local filesystem cache
│   ├── pdfs/                    # Downloaded PDF binaries ({arxiv_id}.pdf)
│   └── summaries/               # Serialized JSON briefing artifacts ({arxiv_id}.json)
├── arxiv_agent/                 # Core Python package
│   ├── __init__.py              # Package entrypoint & version info
│   ├── cli.py                   # Command-line interface & interactive QA REPL
│   ├── config.py                # Environment configuration loader
│   ├── graph.py                 # Stateful agent graph orchestrator (ArxivAgentGraph)
│   ├── state.py                 # Pydantic data schemas & state definitions
│   ├── llm/                     # Multi-provider LLM abstraction layer
│   │   ├── __init__.py
│   │   └── provider.py          # Gemini, Groq, Ollama, OpenAI, OpenRouter, MockProvider
│   ├── nodes/                   # Decoupled graph pipeline nodes
│   │   ├── __init__.py
│   │   ├── query_understanding.py # Node 1: Query normalization & ID extractor
│   │   ├── arxiv_retrieval.py   # Node 2: arXiv Atom XML API harvester
│   │   ├── ranker.py            # Node 3: Semantic candidate ranker & selector
│   │   ├── pdf_fetcher.py       # Node 4a: PDF downloader & cache manager
│   │   ├── pdf_parser.py        # Node 4b: PyMuPDF section & abstract fallback parser
│   │   ├── vector_store.py      # Node 5: Parent-Child chunking & RRF vector index
│   │   ├── summarizer.py        # Node 6: Executive briefing synthesizer & validator
│   │   ├── qa_engine.py         # Node 7: Grounded RAG QA engine & grounding telemetry
│   │   └── comparative_synthesizer.py # Node 8: Multi-paper comparative matrix
│   └── utils/                   # Shared utility modules
│       ├── __init__.py
│       └── formatting.py        # Rich terminal banners, tables & color formatting
├── docs/                        # Architectural & engineering documentation
│   ├── PRD.md                   # Product Requirements Document
│   ├── ARCHITECTURE.md          # System Architecture & State Graph
│   ├── DESIGN.md                # Component & Technical Design Document
│   ├── TEST_PLAN.md             # Automated Test Strategy & Verification Plan
│   ├── SECURITY.md              # Security Policy & Threat Model
│   ├── DECISIONS.md             # Architectural Decision Records (ADRs)
│   └── MEMORY.md                # Operational Memory & System Context (this file)
├── presentation/                # Slide decks & presentation materials
├── tests/                       # Automated test suites (30 tests, 100% offline pass)
│   ├── __init__.py
│   ├── test_query_understanding.py
│   ├── test_arxiv_retrieval.py
│   ├── test_pdf_parser.py
│   ├── test_vector_store.py
│   ├── test_summarizer.py
│   ├── test_qa_engine.py
│   ├── test_graph_flow.py
│   ├── test_failure_handling.py
│   └── test_advanced_features.py
├── generate_demo_video.py       # Script generating 4-minute presentation video
├── main.py                      # Root executable launcher
├── requirements.txt             # Project dependencies
├── README.md                    # Root project documentation
└── .env.example                 # Template for API keys and configuration
```

---

## 3. Configuration & Environment Variables Reference

Configuration values are defined in `arxiv_agent/config.py` and loaded from `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | Active provider: `gemini`, `groq`, `openai`, `ollama`, `openrouter`, `mock` |
| `GEMINI_API_KEY` | `None` | Google AI Studio Gemini API Key |
| `GROQ_API_KEY` | `None` | Groq Cloud API Key |
| `OPENAI_API_KEY` | `None` | OpenAI Platform API Key |
| `OPENROUTER_API_KEY` | `None` | OpenRouter API Key |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model identifier |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |
| `OLLAMA_MODEL` | `llama3.2` | Local Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama local daemon endpoint |
| `ARXIV_MAX_RESULTS` | `5` | Candidate papers retrieved per topic query |
| `REQUEST_TIMEOUT` | `30` | Timeout in seconds for HTTP/PDF requests |
| `CHUNK_SIZE` | `1200` | Target character size for Parent chunks |
| `CHUNK_OVERLAP` | `200` | Overlap character size for Parent chunks |
| `TOP_K_CHUNKS` | `5` | Number of chunks retrieved during RAG QA |
| `CACHE_DIR` | `./.cache_arxiv` | Local filesystem cache path |

---

## 4. Key Developer Workflows & Commands

### 4.1 Running the Agent CLI
```bash
# 1. Topic Search with Interactive QA
python main.py "recent work on KV-cache compression for LLMs" --interactive

# 2. Specific Paper Lookup by arXiv ID
python main.py "2401.12345" -i

# 3. Specific Paper Lookup by arXiv URL
python main.py "https://arxiv.org/abs/1706.03762" -i

# 4. Multi-Paper Comparative Synthesis Matrix
python main.py "efficient attention mechanisms in transformers" --compare

# 5. Export Briefing to Markdown or JSON
python main.py "2401.12345" --output briefing.md
python main.py "2401.12345" --json --output briefing.json

# 6. Completely Offline / Zero-Key Mock Mode
python main.py "2401.12345" --mock --interactive
```

### 4.2 Interactive REPL Commands
- `/briefing`: Display the Executive Briefing.
- `/compare`: Trigger multi-candidate comparative tradeoff analysis.
- `/stats`: Display telemetry (parsed sections, token counts, micro-chunks, fallback flags).
- `/chunks`: Preview indexed parent-child micro-chunks.
- `/export`: Export full session to `qa_export.md`.
- `exit` / `quit`: Terminate interactive session.

### 4.3 Running Automated Tests
```bash
# Run complete test suite (30 tests)
pytest -v tests/

# Run specific test module
pytest -v tests/test_advanced_features.py
```

---

## 5. Operational Quirks & Technical Nuances

1. **arXiv Rate Limiting Politeness**:
   - The arXiv Atom API asks clients to avoid hammering the endpoint with rapid consecutive calls ($>1\text{ req/sec}$).
   - The local `.cache_arxiv/pdfs/` cache guarantees that repeated lookups of the same paper make zero outbound network requests.
2. **PyMuPDF Multi-Column Ordering**:
   - PyMuPDF reads text blocks in standard layout flow. Our regex parser identifies primary section headings across single- and dual-column layouts.
3. **Graceful Fallback Mode (`is_fallback_mode = True`)**:
   - If an arXiv paper has a corrupted PDF binary or blocked download, the agent automatically builds structured sections from the Atom XML abstract. The executive briefing and QA loop remain fully operational.
4. **Anti-Hallucination Refusal String**:
   - Whenever an answer contains insufficient context, the system emits the exact string:
     `"Based on the retrieved sections of this paper, there is not enough information to answer this question."`
   - This sets `grounded = False` and `grounding_level = "REFUSAL"`.

---

## 6. Changelog & Project Evolution

- **v1.0.0**: Initial state graph pipeline with Query Understanding, Atom XML retrieval, PDF parsing, flat vector store, and executive briefing generator.
- **v1.1.0**: Added Parent-Child Hierarchical chunking (~1400 char parents, ~350 char children) and Reciprocal Rank Fusion (RRF: Dense Cosine + Lexical Keyword Overlap).
- **v1.2.0**: Added Quantitative Grounding Confidence scoring ($0.4 + 0.6 \times \text{overlap}$) with `HIGH`, `MEDIUM`, and `REFUSAL` tiers.
- **v1.3.0**: Added Multi-Paper Comparative Synthesis matrix (`--compare` and `/compare`).
- **v1.4.0**: Added full documentation suite in `docs/` (`PRD.md`, `ARCHITECTURE.md`, `DESIGN.md`, `TEST_PLAN.md`, `SECURITY.md`, `DECISIONS.md`, `MEMORY.md`).
