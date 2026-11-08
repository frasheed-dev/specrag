"""SpecRAG Command Line Interface.

Provides CLI commands for:
- specrag benchmark: Runs comparative evaluation on Golden Queries
- specrag ingest: Converts and indexes raw PDF/Markdown specifications
- specrag query: Queries specification knowledge base with Naive, Hybrid, or Reranked modes
"""

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="specrag",
    help="SpecRAG: Structure-Aware Hybrid RAG CLI for Technical Specifications",
    add_completion=False,
)
console = Console()


@app.command()
def benchmark(
    dataset: str = typer.Option(
        "local",
        "--dataset",
        "-d",
        help="Benchmark dataset to run: 'local', 't2' (T2-RAGBench), 'ibm' (IBM TechReport), or 'all'",
    ),
    run_llm: bool = typer.Option(
        False,
        "--run-llm",
        "-l",
        help="Run LLM generation and qualitative faithfulness checks",
    ),
) -> None:
    """Run the evaluation benchmark comparing Naive vs Production Hybrid RAG."""
    console.print(
        Panel.fit(
            "[bold cyan]SpecRAG Multi-Corpus Benchmark Harness[/bold cyan]\n"
            f"Evaluating Dataset: [bold yellow]{dataset.upper()}[/bold yellow] | Naive Fixed-Window vs. Structure-Aware Hybrid + Reranker",
            border_style="cyan",
        )
    )

    from specrag.evaluator import BenchmarkRunner

    runner = BenchmarkRunner()

    datasets_to_run = ["local", "t2", "ibm"] if dataset.lower() == "all" else [dataset.lower()]

    summary_table = Table(
        title="SpecRAG Multi-Benchmark Performance Summary",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Benchmark Suite", style="cyan", width=25)
    summary_table.add_column("Pipeline Architecture", width=35)
    summary_table.add_column("Hit Rate @ 5", justify="right")
    summary_table.add_column("MRR", justify="right")
    summary_table.add_column("Avg Latency", justify="right")

    for ds in datasets_to_run:
        console.print(
            f"\n[cyan]Running benchmark on dataset suite: [bold]{ds.upper()}[/bold]...[/cyan]"
        )
        results = runner.run_benchmark(dataset=ds, run_llm_generation=run_llm)
        naive = results["naive"]
        hybrid = results["hybrid"]

        summary_table.add_row(
            ds.upper(),
            "Phase A: Naive Baseline",
            f"{naive['hit_rate_at_5'] * 100:.1f}%",
            f"{naive['mrr']:.3f}",
            f"{naive['avg_latency_ms']:.1f} ms",
        )
        summary_table.add_row(
            ds.upper(),
            "[bold green]Phase B: Production Hybrid + Rerank[/bold green]",
            f"[bold green]{hybrid['hit_rate_at_5'] * 100:.1f}%[/bold green]",
            f"[bold green]{hybrid['mrr']:.3f}[/bold green]",
            f"{hybrid['avg_latency_ms']:.1f} ms",
        )

    console.print("\n")
    console.print(summary_table)
    console.print("\n[bold green]Results updated in BENCHMARK_RESULTS.md[/bold green]\n")


@app.command()
def ingest(
    file_path: Path = typer.Argument(..., help="Path to input PDF or Markdown specification"),
    output_dir: Path = typer.Option(
        Path("data/specs"),
        "--output-dir",
        "-o",
        help="Destination directory for converted markdown files",
    ),
) -> None:
    """Convert and ingest a specification document (PDF -> Markdown with preserved tables)."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    from specrag.ingestion.converter import SpecDocumentConverter

    converter = SpecDocumentConverter()
    console.print(f"[cyan]Ingesting and converting:[/cyan] {file_path.name}...")

    t0 = time.perf_counter()
    out_file = converter.convert_pdf(file_path, output_dir=output_dir)
    elapsed = (time.perf_counter() - t0) * 1000

    console.print(
        Panel.fit(
            f"[bold green]Ingestion Complete![/bold green]\n"
            f"Input: {file_path}\n"
            f"Output Markdown: {out_file}\n"
            f"Time elapsed: {elapsed:.1f} ms",
            border_style="green",
        )
    )


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Specification query or SKU/clause lookup"),
    mode: str = typer.Option(
        "hybrid_rerank",
        "--mode",
        "-m",
        help="Retrieval mode: 'naive', 'hybrid', or 'hybrid_rerank'",
    ),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of retrieved chunks"),
    specs_dir: Path = typer.Option(
        Path("data/specs"),
        "--specs-dir",
        "-d",
        help="Directory containing markdown specifications",
    ),
    generate: bool = typer.Option(
        False,
        "--generate",
        "-g",
        help="Synthesize structured answer using LLM (requires API key in environment)",
    ),
) -> None:
    """Query specifications using Naive, Hybrid, or Hybrid+Rerank retrieval pipelines."""
    if not specs_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Specs directory does not exist: {specs_dir}")
        raise typer.Exit(code=1)

    spec_files = list(specs_dir.glob("*.md"))
    if not spec_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No markdown specs found in {specs_dir}")
        raise typer.Exit(code=1)

    documents = {f.name: f.read_text(encoding="utf-8") for f in spec_files}

    console.print(
        f"[bold cyan]Query:[/bold cyan] {query_text} [dim](Mode: {mode}, Top-K: {top_k})[/dim]"
    )

    if mode == "naive":
        from specrag.chunkers import FixedWindowChunker
        from specrag.retriever import QdrantDenseRetriever

        chunker = FixedWindowChunker(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        for doc_id, text in documents.items():
            all_chunks.extend(chunker.chunk_document(text, doc_id))
        retriever = QdrantDenseRetriever(collection_name="cli_naive", recreate_collection=True)
        retriever.index_chunks(all_chunks)
        hits = retriever.retrieve(query_text, top_k=top_k)
        retrieved_chunks = [c for c, _ in hits]
    else:
        from specrag.chunkers import StructureAwareMarkdownChunker
        from specrag.retriever import (
            BM25SparseRetriever,
            CrossEncoderReranker,
            HybridRRFRetriever,
            QdrantDenseRetriever,
        )

        chunker = StructureAwareMarkdownChunker(max_prose_chunk_size=1000)
        all_chunks = []
        for doc_id, text in documents.items():
            all_chunks.extend(chunker.chunk_document(text, doc_id))

        dense = QdrantDenseRetriever(collection_name="cli_hybrid", recreate_collection=True)
        dense.index_chunks(all_chunks)

        sparse = BM25SparseRetriever()
        sparse.index_chunks(all_chunks)

        hybrid = HybridRRFRetriever(dense_retriever=dense, sparse_retriever=sparse)

        if mode == "hybrid_rerank":
            candidates = hybrid.retrieve(query_text, top_k=max(20, top_k * 3))
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(query_text, [c for c, _ in candidates], top_k=top_k)
            retrieved_chunks = [c for c, _ in reranked]
        else:
            hits = hybrid.retrieve(query_text, top_k=top_k)
            retrieved_chunks = [c for c, _ in hits]

    # Display retrieved chunks
    for i, c in enumerate(retrieved_chunks, start=1):
        table_flag = "[bold red][TABLE][/bold red]" if c.is_table else "[blue][PROSE][/blue]"
        prefix = f" | {c.breadcrumb_prefix}" if c.breadcrumb_prefix else ""
        console.print(
            Panel(
                f"[bold yellow]Chunk #{i}[/bold yellow] {table_flag} [dim]{c.doc_id}{prefix}[/dim]\n\n"
                f"{c.text[:400]}..."
                if len(c.text) > 400
                else c.text,
                title=f"Result {i} (ID: {c.chunk_id})",
                border_style="blue",
            )
        )

    if generate:
        console.print("\n[cyan]Synthesizing structured answer with LLM...[/cyan]")
        try:
            from specrag.generator import SpecAnswerGenerator

            gen = SpecAnswerGenerator()
            response = gen.generate(query_text, retrieved_chunks)
            console.print(
                Panel.fit(
                    f"[bold green]Answer:[/bold green]\n{response.answer}\n\n"
                    f"[dim]Confidence: {response.confidence_score:.2f} | Reasoning: {response.reasoning}[/dim]\n"
                    f"[bold cyan]Citations:[/bold cyan] {', '.join(str(c) for c in response.citations)}",
                    title="SpecRAG Synthesized Answer",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(f"[bold red]LLM Generation failed or no API key set:[/bold red] {e}")


if __name__ == "__main__":
    app()
