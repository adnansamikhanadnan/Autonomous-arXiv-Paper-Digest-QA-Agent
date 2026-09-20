# Automated Test Plan & Quality Assurance Strategy

## Project: Autonomous arXiv Paper Digest & Grounded QA Agent

---

## 1. Testing Philosophy & Objectives

The testing suite for the Autonomous arXiv Agent is engineered according to three core tenets:
1. **Zero External Flakiness (100% Offline Determinism)**: Automated tests must run and pass without requiring an active internet connection, external API keys, or live arXiv network calls.
2. **Comprehensive Boundary Coverage**: Every pipeline stage, edge case (e.g., legacy arXiv ID formats, corrupted PDFs), and fallback mode must be rigorously validated.
3. **Strict Schema & Guardrail Compliance**: Tests must assert that output briefings satisfy all Pydantic constraints (including the mandatory *Limitations* section) and that adversarial QA prompts trigger factual refusals.

---

## 2. Test Architecture & Pyramid

```
                ▲
               / \
              /   \
             / E2E \       End-to-End Graph Flow Tests
            / Graph \      (test_graph_flow.py)
           /─────────\
          /   Fault   \    Failure Injection & Fallbacks
         /  Handling   \   (test_failure_handling.py)
        /───────────────\
       /   Integration   \  Advanced Features & Multi-Node
      /   & Multi-Node    \ (test_advanced_features.py, test_qa_engine.py)
     /─────────────────────\
    /      Unit Tests       \ Component-Level Node Tests
   /  (Nodes 1, 2, 3, 4, 5, 6)\ (query, retrieval, parser, vector, summarizer)
  /───────────────────────────\
```

---

## 3. Test Suites Inventory

The repository contains 9 dedicated test suites comprising 30 automated tests:

| Test Module | Focus Area | Test Count | Key Invariants Verified |
| :--- | :--- | :--- | :--- |
| [`test_query_understanding.py`](file:///d:/Anti%20gravity/Project/tests/test_query_understanding.py) | Query Parsing & ID Extraction | 6 | Modern arXiv IDs, versioned IDs, legacy slash IDs, URL normalization, topic cleaning |
| [`test_arxiv_retrieval.py`](file:///d:/Anti%20gravity/Project/tests/test_arxiv_retrieval.py) | Atom XML Harvester | 3 | XML parsing, author list aggregation, category tag extraction, URL construction |
| [`test_pdf_parser.py`](file:///d:/Anti%20gravity/Project/tests/test_pdf_parser.py) | PDF & Section Extraction | 3 | Section regex detection, page span calculation, abstract fallback parser |
| [`test_vector_store.py`](file:///d:/Anti%20gravity/Project/tests/test_vector_store.py) | Chunking & Vector Search | 4 | Parent-child splitting, sliding-window overlap, cosine similarity, lexical boost |
| [`test_summarizer.py`](file:///d:/Anti%20gravity/Project/tests/test_summarizer.py) | Executive Briefing Generation | 3 | Pydantic schema validation, mandatory limitations constraint, Markdown rendering |
| [`test_qa_engine.py`](file:///d:/Anti%20gravity/Project/tests/test_qa_engine.py) | Grounded RAG & Citations | 2 | Top-k chunk retrieval, section/page citations, out-of-scope refusal |
| [`test_graph_flow.py`](file:///d:/Anti%20gravity/Project/tests/test_graph_flow.py) | End-to-End State Graph | 2 | Full pipeline on topic queries and direct arXiv IDs |
| [`test_failure_handling.py`](file:///d:/Anti%20gravity/Project/tests/test_failure_handling.py) | Fault Injection & Resilience | 3 | Zero search results recovery, broken PDF URLs, corrupt binary fallback |
| [`test_advanced_features.py`](file:///d:/Anti%20gravity/Project/tests/test_advanced_features.py) | Advanced RAG & Synthesis | 4 | Parent-child chunk linking, RRF ranking, grounding confidence score, comparative matrix |

---

## 4. Test Case Specifications

### 4.1 Query Understanding (`test_query_understanding.py`)
- **`test_extract_modern_arxiv_id`**: Confirms `2401.12345` is extracted and tagged as `SPECIFIC_PAPER`.
- **`test_extract_versioned_arxiv_id`**: Confirms `2401.12345v2` is properly normalized to `2401.12345`.
- **`test_extract_legacy_arxiv_id`**: Confirms old-style IDs (`cs/0101001`, `math.GT/0309136`) are recognized.
- **`test_extract_arxiv_url`**: Parses `https://arxiv.org/abs/1706.03762` and `https://arxiv.org/pdf/1706.03762.pdf`.
- **`test_clean_topic_query`**: Verifies that conversational prefixes (*"can you find recent papers on..."*) are stripped.
- **`test_intent_classification`**: Ensures clear routing between `TOPIC_SEARCH` and `SPECIFIC_PAPER`.

### 4.2 PDF Parser & Section Extractor (`test_pdf_parser.py`)
- **`test_section_regex_detection`**: Verifies regex captures *Introduction*, *Methodology*, *Results*, and *Limitations*.
- **`test_hyphenation_and_clean_text`**: Ensures line-break hyphenations (e.g., `trans-\nformer` $\rightarrow$ `transformer`) are merged cleanly.
- **`test_abstract_fallback_mode`**: Validates that when PDF binary is absent, `parse_from_metadata` builds structured sections from the Atom abstract without raising an exception.

### 4.3 Hierarchical Vector Index & RRF (`test_vector_store.py` & `test_advanced_features.py`)
- **`test_parent_child_chunking`**: Confirms parent chunks (~1400 chars) are created with associated child micro-chunks (~350 chars) and that `parent_id` foreign keys match.
- **`test_rrf_vector_search`**: Confirms that Reciprocal Rank Fusion ranks relevant chunks at rank 1 based on both cosine similarity and lexical token overlap.
- **`test_calculate_grounding_score`**: Asserts that answers matching source context score $\ge 0.70$ (`HIGH`) and that refusal answers return score $0.0$ (`REFUSAL`).

### 4.4 Summarizer & Schema Validation (`test_summarizer.py`)
- **`test_executive_briefing_schema`**: Ensures all 6 required fields are populated and typed correctly.
- **`test_mandatory_limitations`**: Asserts that `briefing.limitations` contains at least 1 explicit limitation item.
- **`test_markdown_formatting`**: Asserts that rendered Markdown output includes all standard section headings and links.

### 4.5 Grounded QA & Anti-Hallucination (`test_qa_engine.py`)
- **`test_grounded_answer_with_citations`**: Confirms that valid technical questions return answers citing `[Section, Page]`.
- **`test_anti_hallucination_refusal`**: Passes an adversarial out-of-domain prompt and asserts that the engine emits a standardized refusal string with `grounded = False`.

### 4.6 Fault Injection & Resilience (`test_failure_handling.py`)
- **`test_zero_search_results`**: Asserts pipeline transitions to `FAILED` with a user-friendly error message and keyword suggestions rather than an unhandled traceback.
- **`test_broken_pdf_download`**: Simulates HTTP 404/500 errors during download and verifies automatic switch to `is_fallback_mode = True`.
- **`test_corrupted_pdf_stream`**: Feeds randomized binary noise to the PDF parser and confirms clean fallback to abstract sections.

---

## 5. Test Execution Procedures

### 5.1 Running the Complete Suite
```bash
# Run all tests with verbose output
pytest -v tests/

# Run with coverage report
pytest --cov=arxiv_agent tests/
```

### 5.2 Running Specific Test Suites
```bash
# Test only failure handling
pytest -v tests/test_failure_handling.py

# Test Parent-Child RRF and Grounding Score
pytest -v tests/test_advanced_features.py
```

### 5.3 Continuous Integration (CI) Invariants
1. All tests must execute in under 10 seconds total.
2. 0 external network requests allowed during test run.
3. Test suite must exit with return code `0`.
