"""CLI Interface and Interactive QA REPL with Advanced Telemetry for arXiv Agent."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from arxiv_agent.config import Config
from arxiv_agent.state import AgentStatus, ExecutiveBriefing
from arxiv_agent.llm.provider import get_llm_provider, MockProvider
from arxiv_agent.graph import ArxivAgentGraph
from arxiv_agent.nodes.summarizer import format_briefing_markdown
from arxiv_agent.utils.formatting import (
    render_banner,
    render_stage_progress,
    render_candidates_table,
    render_executive_briefing,
    render_qa_turn,
    render_comparative_matrix,
)


def on_graph_step(step_name: str, state) -> None:
    """Callback to render graph execution progress live to terminal."""
    if step_name == "query_understanding_start":
        render_stage_progress("Query Understanding", "Parsing intent and extracting arXiv ID/topics...")
    elif step_name == "query_understanding_end":
        render_stage_progress("Query Understanding", f"Intent: {state.intent.value.upper()}", "success")

    elif step_name == "arxiv_retrieval_start":
        render_stage_progress("arXiv Retrieval", "Calling official arXiv Atom API...")
    elif step_name == "arxiv_retrieval_end":
        if state.candidate_papers:
            render_stage_progress("arXiv Retrieval", f"Retrieved {len(state.candidate_papers)} paper(s)", "success")
            render_candidates_table(state.candidate_papers)
        else:
            render_stage_progress("arXiv Retrieval", "No candidate papers found", "warning")

    elif step_name == "comparison_start":
        render_stage_progress("Comparative Synthesis", "Analyzing landscape tradeoffs across candidates...")
    elif step_name == "comparison_end":
        render_stage_progress("Comparative Synthesis", "Comparative matrix generated", "success")

    elif step_name == "ranking_start":
        render_stage_progress("Paper Selection", "Evaluating semantic match and technical relevance...")
    elif step_name == "ranking_end":
        if state.selected_paper:
            render_stage_progress(
                "Paper Selection",
                f"Selected '{state.selected_paper.title}' (arXiv:{state.selected_paper.arxiv_id})",
                "success",
            )

    elif step_name == "pdf_fetch_start":
        render_stage_progress("PDF Acquisition", "Downloading PDF and verifying integrity...")
    elif step_name == "pdf_fetch_end":
        if state.pdf_path:
            render_stage_progress("PDF Acquisition", f"PDF cached at {state.pdf_path}", "success")
        else:
            render_stage_progress("PDF Acquisition", "Download failed, using abstract fallback", "warning")

    elif step_name == "pdf_parse_start":
        render_stage_progress("PDF Parsing", "Extracting structured sections and hierarchy...")
    elif step_name == "pdf_parse_end":
        render_stage_progress(
            "PDF Parsing",
            f"Extracted {len(state.parsed_sections)} section(s) ({len(state.full_text)} chars)",
            "success",
        )

    elif step_name == "vector_indexing_start":
        render_stage_progress("Vector Indexing", "Building Parent-Child chunks and RRF index...")
    elif step_name == "vector_indexing_end":
        render_stage_progress("Vector Indexing", f"Indexed {len(state.chunks)} micro-chunks in local RRF vector DB", "success")

    elif step_name == "summarizing_start":
        render_stage_progress("Summarization", "Synthesizing executive briefing artifact...")
    elif step_name == "summarizing_end":
        render_stage_progress("Summarization", "Executive briefing generated", "success")

    elif step_name == "qa_setup_end":
        render_stage_progress("QA System", "Parent-Child RRF QA pipeline active with Grounding Telemetry", "success")


def run_interactive_qa(graph: ArxivAgentGraph) -> None:
    """Interactive QA loop allowing the user to ask follow-up questions and inspect telemetry."""
    print("\n" + "=" * 60)
    print("💬 INTERACTIVE QA MODE (Parent-Child RAG + Grounding Telemetry)")
    print("Commands: '/compare', '/stats', '/chunks', '/briefing', '/export', 'exit'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\n📝 Ask a question > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting QA session. Goodbye!\n")
                break

            if user_input.lower() == "/briefing":
                if graph.current_state and graph.current_state.executive_briefing:
                    render_executive_briefing(graph.current_state.executive_briefing)
                continue

            if user_input.lower() == "/compare":
                if graph.current_state and graph.current_state.candidate_papers:
                    analysis = graph.compare_candidates()
                    render_comparative_matrix(analysis)
                else:
                    print("No candidate papers in state to compare.")
                continue

            if user_input.lower() == "/stats":
                if graph.current_state:
                    state = graph.current_state
                    print("\n--- Pipeline Telemetry & Stats ---")
                    print(f"Target Paper: {state.selected_paper.title if state.selected_paper else 'N/A'}")
                    print(f"Sections Parsed: {len(state.parsed_sections)}")
                    print(f"Total Text Size: {len(state.full_text)} characters")
                    print(f"Indexed Micro-Chunks: {len(state.chunks)}")
                    print(f"QA Exchanges: {len(state.qa_history) // 2}")
                    print(f"Fallback Mode Active: {state.is_fallback_mode}")
                    print("----------------------------------\n")
                continue

            if user_input.lower() == "/chunks":
                if graph.current_state:
                    print(f"\nIndexed Chunks Preview (Top 5 of {len(graph.current_state.chunks)}):")
                    for i, c in enumerate(graph.current_state.chunks[:5], 1):
                        print(f"[{i}] {c.chunk_id} | Section: {c.section_title} | Page {c.page_number} | Tokens: {c.token_count}")
                        print(f"    Excerpt: {c.text[:100]}...\n")
                continue

            if user_input.lower() == "/export":
                export_path = Path("qa_export.md")
                if graph.current_state:
                    with open(export_path, "w", encoding="utf-8") as f:
                        if graph.current_state.executive_briefing:
                            f.write(format_briefing_markdown(graph.current_state.executive_briefing))
                        if graph.current_state.comparative_analysis:
                            f.write("\n\n# Multi-Paper Comparative Matrix\n\n")
                            f.write(f"Topic: {graph.current_state.comparative_analysis.topic}\n\n")
                            for row in graph.current_state.comparative_analysis.papers:
                                f.write(f"- **{row.title}** (arXiv:{row.arxiv_id}): {row.core_approach}\n")
                        f.write("\n\n# QA Interaction History\n\n")
                        for msg in graph.current_state.qa_history:
                            f.write(f"### {msg.role.capitalize()}:\n{msg.content}\n\n")
                    print(f"✔ Exported briefing and QA history to {export_path.resolve()}")
                continue

            # Execute QA query
            qa_response = graph.ask(user_input)
            render_qa_turn(qa_response)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Autonomous arXiv Paper Digest & QA Agent - State Graph Pipeline"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Research topic or arXiv ID / URL (e.g. '2401.12345' or 'KV-cache compression')",
    )
    parser.add_argument("-q", "--query-arg", dest="query_flag", help="Alternative flag for query")
    parser.add_argument(
        "-p", "--provider",
        choices=["gemini", "groq", "openai", "ollama", "openrouter", "mock"],
        default=None,
        help="LLM provider (default: auto-detected from .env)",
    )
    parser.add_argument("-m", "--model", default=None, help="Model name override")
    parser.add_argument("-o", "--output", default=None, help="File path to save briefing (markdown or json)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON briefing to stdout")
    parser.add_argument("--compare", action="store_true", help="Generate multi-paper comparative synthesis matrix")
    parser.add_argument("--offline", "--mock", action="store_true", help="Force offline mock provider")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive QA loop after briefing")

    args = parser.parse_args()

    # Determine query
    query_text = args.query or args.query_flag
    if not query_text:
        render_banner()
        query_text = input("Enter a research topic or arXiv ID/URL: ").strip()
        if not query_text:
            print("Error: No query provided. Exiting.")
            sys.exit(1)
        args.interactive = True
    else:
        render_banner()

    # Determine provider
    if args.offline:
        llm = MockProvider()
    else:
        llm = get_llm_provider(provider_name=args.provider, model_name=args.model)

    # Initialize Graph
    graph = ArxivAgentGraph(llm_provider=llm, on_step_callback=on_graph_step)

    # Run pipeline
    print(f"\n🚀 Running Autonomous arXiv Agent Pipeline on: '{query_text}'\n")
    state = graph.run(query_text, generate_comparison=args.compare)

    if state.status == AgentStatus.FAILED or not state.executive_briefing:
        print("\n❌ Pipeline failed:")
        for err in state.errors:
            print(f" - {err}")
        sys.exit(1)

    # Render Comparative Matrix if generated
    if state.comparative_analysis:
        render_comparative_matrix(state.comparative_analysis)

    # Render Executive Briefing
    if args.json:
        print(state.executive_briefing.model_dump_json(indent=2))
    else:
        render_executive_briefing(state.executive_briefing)

    # Save artifact if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(state.executive_briefing.model_dump_json(indent=2))
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(format_briefing_markdown(state.executive_briefing))
        print(f"\n✔ Executive Briefing saved to: {out_path.resolve()}")

    # Launch interactive QA loop if requested
    if args.interactive:
        run_interactive_qa(graph)


if __name__ == "__main__":
    main()
