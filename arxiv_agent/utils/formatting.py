"""Rich UI formatting utilities for CLI output and grounding telemetry."""

import sys
from typing import List, Optional
from arxiv_agent.state import ExecutiveBriefing, PaperMetadata, QAMessage, ComparativeAnalysis

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    # pyrefly: ignore [missing-import]
    from rich.console import Console
    # pyrefly: ignore [missing-import]
    from rich.panel import Panel
    # pyrefly: ignore [missing-import]
    from rich.table import Table
    # pyrefly: ignore [missing-import]
    from rich.markdown import Markdown
    # pyrefly: ignore [missing-import]
    from rich.text import Text
    console = Console(legacy_windows=False, force_terminal=False)
    HAS_RICH = True
except Exception:
    HAS_RICH = False
    console = None


def render_banner() -> None:
    """Render welcome banner."""
    if HAS_RICH and console:
        title_text = Text("🔬 Autonomous arXiv Paper Digest & QA Agent", style="bold cyan")
        subtitle_text = Text("Hierarchical RAG • RRF Hybrid Search • Grounded QA Telemetry", style="dim white")
        console.print(Panel(Text.assemble(title_text, "\n", subtitle_text), border_style="bright_blue"))
    else:
        print("==================================================")
        print(" Autonomous arXiv Paper Digest & QA Agent")
        print(" Hierarchical RAG • RRF Hybrid Search • Grounded QA Telemetry")
        print("==================================================")


def render_stage_progress(stage_name: str, description: str, status: str = "running") -> None:
    """Print stage execution updates."""
    if HAS_RICH and console:
        if status == "running":
            console.print(f"[bold cyan]▶ {stage_name}:[/bold cyan] [dim]{description}[/dim]")
        elif status == "success":
            console.print(f"[bold green]✔ {stage_name}:[/bold green] {description}")
        elif status == "warning":
            console.print(f"[bold yellow]▲ {stage_name}:[/bold yellow] {description}")
        elif status == "error":
            console.print(f"[bold red]✖ {stage_name}:[/bold red] {description}")
    else:
        prefix = "▶" if status == "running" else ("✔" if status == "success" else "▲")
        print(f"{prefix} {stage_name}: {description}")


def render_candidates_table(candidates: List[PaperMetadata]) -> None:
    """Render table of candidate papers found on arXiv."""
    if HAS_RICH and console:
        table = Table(title="Candidate Papers Found on arXiv", border_style="blue", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("arXiv ID", style="bold green", width=14)
        table.add_column("Title", style="white", width=40)
        table.add_column("Authors", style="dim", width=20)
        table.add_column("Published", style="magenta", width=12)

        for i, p in enumerate(candidates, 1):
            authors_preview = ", ".join(p.authors[:2]) + ("..." if len(p.authors) > 2 else "")
            table.add_row(str(i), p.arxiv_id, p.title[:60], authors_preview, p.published)

        console.print(table)
    else:
        print("\n--- Candidate Papers Found on arXiv ---")
        for i, p in enumerate(candidates, 1):
            print(f"[{i}] arXiv:{p.arxiv_id} | {p.title} ({p.published})")
        print("---------------------------------------\n")


def render_executive_briefing(briefing: ExecutiveBriefing) -> None:
    """Render the structured executive briefing artifact."""
    if HAS_RICH and console:
        header_text = (
            f"[bold white]{briefing.title}[/bold white]\n\n"
            f"[dim]Authors:[/dim] {', '.join(briefing.authors)}\n"
            f"[dim]arXiv ID:[/dim] [bold cyan]{briefing.arxiv_id}[/bold cyan]  |  "
            f"[dim]Published:[/dim] [magenta]{briefing.publish_date}[/magenta]  |  "
            f"[dim]Link:[/dim] [blue underline]{briefing.link}[/blue underline]"
        )
        console.print(Panel(header_text, title="📄 EXECUTIVE BRIEFING", border_style="bright_green"))

        # Plain English Summary
        console.print(
            Panel(
                f"[italic]{briefing.plain_english_summary}[/italic]",
                title="📌 Plain-English Summary (Why This Paper Matters)",
                border_style="cyan",
            )
        )

        # Problem Statement
        console.print(
            Panel(
                briefing.problem_statement,
                title="🎯 Problem Statement",
                border_style="yellow",
            )
        )

        # Methodology & Results
        method_str = "\n".join([f"• [bold white]{m}[/bold white]" for m in briefing.method_approach])
        console.print(Panel(method_str, title="🔬 Method & Approach", border_style="blue"))

        results_str = "\n".join([f"• [green]{r}[/green]" for r in briefing.key_results_claims])
        console.print(Panel(results_str, title="📊 Key Results & Empirical Claims", border_style="green"))

        # Explicit Limitations (Highlighted)
        limits_str = "\n".join([f"• [bold red]{l}[/bold red]" for l in briefing.limitations])
        console.print(
            Panel(
                limits_str,
                title="⚠️ Explicit Limitations & Boundary Constraints (Mandatory)",
                border_style="red",
            )
        )

        # Suggested Questions
        questions_str = "\n".join(
            [f"[cyan]{i}.[/cyan] {q}" for i, q in enumerate(briefing.suggested_follow_up_questions, 1)]
        )
        console.print(
            Panel(
                questions_str,
                title="💡 Suggested Follow-up Questions for Investigation",
                border_style="magenta",
            )
        )
    else:
        print("\n" + "=" * 60)
        print(f"EXECUTIVE BRIEFING: {briefing.title}")
        print(f"Authors: {', '.join(briefing.authors)}")
        print(f"arXiv ID: {briefing.arxiv_id} | Published: {briefing.publish_date}")
        print(f"Link: {briefing.link}")
        print("=" * 60)
        print(f"\n[PLAIN-ENGLISH SUMMARY]\n{briefing.plain_english_summary}")
        print(f"\n[PROBLEM STATEMENT]\n{briefing.problem_statement}")
        print("\n[METHOD & APPROACH]")
        for m in briefing.method_approach:
            print(f" - {m}")
        print("\n[KEY RESULTS & CLAIMS]")
        for r in briefing.key_results_claims:
            print(f" - {r}")
        print("\n[EXPLICIT LIMITATIONS]")
        for l in briefing.limitations:
            print(f" - {l}")
        print("\n[SUGGESTED QUESTIONS]")
        for q in briefing.suggested_follow_up_questions:
            print(f" - {q}")
        print("=" * 60 + "\n")


def render_qa_turn(qa_msg: QAMessage) -> None:
    """Render an individual QA answer with quantitative grounding badge and citation table."""
    if HAS_RICH and console:
        # Grounding Badge
        conf_pct = int(qa_msg.grounding_confidence * 100)
        if qa_msg.grounding_level == "HIGH":
            badge = f"[bold white on green]  ✔ GROUNDED ({conf_pct}% Confidence)  [/bold white on green]"
        elif qa_msg.grounding_level == "MEDIUM":
            badge = f"[bold white on yellow]  ▲ PARTIAL ({conf_pct}% Confidence)  [/bold white on yellow]"
        else:
            badge = f"[bold white on red]  ✖ REFUSAL (Ungrounded / Out of Scope)  [/bold white on red]"

        console.print(badge)
        console.print(Panel(qa_msg.content, title="🤖 Agent Answer", border_style="cyan"))

        if qa_msg.citations:
            cit_table = Table(title="Retrieved Chunk Grounding & Sources (RRF + Parent Context)", border_style="dim white", show_lines=False)
            cit_table.add_column("Section", style="bold cyan", width=25)
            cit_table.add_column("Page", style="magenta", width=6)
            cit_table.add_column("RRF Score", style="yellow", width=10)
            cit_table.add_column("Excerpt Preview", style="dim", width=45)

            for cit in qa_msg.citations:
                cit_table.add_row(
                    cit["section"],
                    str(cit["page"]),
                    str(cit["score"]),
                    cit["preview"],
                )
            console.print(cit_table)
    else:
        print(f"\n[Agent Answer] (Confidence: {int(qa_msg.grounding_confidence * 100)}% | {qa_msg.grounding_level}):\n{qa_msg.content}\n")
        if qa_msg.citations:
            print("Sources:")
            for cit in qa_msg.citations:
                print(f" - Section: {cit['section']} | Page {cit['page']} | Score {cit['score']}")
        print("")


def render_comparative_matrix(analysis: ComparativeAnalysis) -> None:
    """Render a multi-paper comparative synthesis matrix."""
    if HAS_RICH and console:
        table = Table(title=f"📊 Multi-Paper Comparative Matrix: '{analysis.topic}'", border_style="cyan", show_lines=True)
        table.add_column("arXiv ID", style="bold green", width=14)
        table.add_column("Title", style="white", width=30)
        table.add_column("Core Mechanism", style="cyan", width=30)
        table.add_column("Key Advantages", style="green", width=25)
        table.add_column("Primary Limitations", style="red", width=25)

        for row in analysis.papers:
            table.add_row(
                row.arxiv_id,
                row.title[:50],
                row.core_approach,
                row.key_advantages,
                row.primary_limitations,
            )
        console.print(table)

        # Synthesis Summary Box
        summary_text = (
            f"[bold white]Synthesis of Tradeoffs:[/bold white]\n{analysis.synthesis_summary}\n\n"
            f"[dim]Recommended Reading Order:[/dim] {' → '.join(analysis.recommended_reading_order)}"
        )
        console.print(Panel(summary_text, title="🔍 Landscape Synthesis", border_style="bright_magenta"))
    else:
        print("\n" + "=" * 60)
        print(f"COMPARATIVE MATRIX: {analysis.topic}")
        print("=" * 60)
        for row in analysis.papers:
            print(f"arXiv:{row.arxiv_id} | {row.title}")
            print(f" - Approach: {row.core_approach}")
            print(f" - Advantages: {row.key_advantages}")
            print(f" - Limitations: {row.primary_limitations}\n")
        print(f"[Synthesis]\n{analysis.synthesis_summary}")
        print(f"[Recommended Reading Order]: {' -> '.join(analysis.recommended_reading_order)}")
        print("=" * 60 + "\n")
