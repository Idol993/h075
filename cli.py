from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from io_parser import read_sequences, write_fasta, write_newick, NewickNode
from aligner import AlignmentEngine
from matrix import compute_similarity_matrix, compute_distance_matrix
from phylogeny import (
    build_nj_tree,
    build_upgma_tree,
    collect_leaf_names,
    compute_bootstrap_support,
    assign_bootstrap_to_nodes,
)
from visualizer import render_heatmap, render_tree_ascii, render_bar_chart
from stats import compute_sequence_stats, render_stats


console = Console()


@click.group()
@click.version_option("1.0.0")
def cli():
    """BioSeq Toolkit - Local sequence alignment & phylogenetic tree CLI"""
    pass


@cli.command()
@click.option("-i", "--input", "input_file", required=True, help="Input FASTA/FASTQ file")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["needle", "water"]),
    default="needle",
    help="Alignment mode: needle (Needleman-Wunsch) or water (Smith-Waterman)",
)
@click.option("-o", "--output", "output_file", default=None, help="Output aligned FASTA file")
@click.option("--checkpoint-dir", default=None, help="Directory for checkpoint files")
@click.option("--resume", is_flag=True, default=False, help="Resume from checkpoint")
def align(input_file, mode, output_file, checkpoint_dir, resume):
    """Pairwise align sequences and generate similarity matrix heatmap."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    names = [r.id for r in records]
    sequences = [str(r.seq) for r in records]
    n = len(records)
    console.print(f"[green]Loaded {n} sequences[/green]")

    if n > 2000:
        console.print(f"[yellow]Large dataset ({n} sequences). Batch processing enabled.[/yellow]")
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing similarities...", total=n * (n - 1) // 2)

        def on_progress(done, total):
            progress.update(task, completed=done, total=total)

        sim_mat = compute_similarity_matrix(
            sequences,
            names,
            mode=mode,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            progress_callback=on_progress,
        )

    console.print()
    console.print("[bold green]Similarity matrix computed![/bold green]")
    render_heatmap(sim_mat.matrix, names, console=console)

    if output_file:
        engine = AlignmentEngine(mode=mode)
        aligned_pairs = engine.align_multiple(sequences, names)
        write_fasta(
            [__make_record(name, seq) for name, seq in aligned_pairs],
            output_file,
        )
        console.print(f"[green]Aligned sequences saved to {output_file}[/green]")


@cli.command()
@click.option("-i", "--input", "input_file", required=True, help="Input aligned/unaligned FASTA")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["nj", "upgma"]),
    default="nj",
    help="Tree construction method: nj (Neighbor-Joining) or upgma",
)
@click.option("-o", "--output", "output_file", default=None, help="Output Newick file")
@click.option(
    "--distance-mode",
    type=click.Choice(["needle", "water"]),
    default="needle",
    help="Distance computation alignment mode (if input is unaligned FASTA)",
)
@click.option(
    "-b",
    "--bootstrap",
    "bootstrap_replicates",
    type=int,
    default=100,
    help="Bootstrap replicates (0 to disable, default 100)",
)
@click.option("--checkpoint-dir", default=None, help="Directory for checkpoint files")
@click.option("--resume", is_flag=True, default=False, help="Resume from checkpoint")
def tree(input_file, method, output_file, distance_mode, bootstrap_replicates, checkpoint_dir, resume):
    """Build phylogenetic tree (NJ/UPGMA) with bootstrap and render ASCII preview."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    names = [r.id for r in records]
    sequences = [str(r.seq) for r in records]
    n = len(records)
    console.print(f"[green]Loaded {n} sequences[/green]")

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing distance matrix...", total=n * (n - 1) // 2)

        def on_progress(done, total):
            progress.update(task, completed=done, total=total)

        dist_matrix, ordered_names = compute_distance_matrix(
            sequences,
            names,
            mode=distance_mode,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            progress_callback=on_progress,
        )

    console.print()
    console.print("[bold green]Distance matrix computed![/bold green]")

    if method == "nj":
        console.print("[cyan]Building Neighbor-Joining tree...[/cyan]")
        root = build_nj_tree(dist_matrix, ordered_names)
    else:
        console.print("[cyan]Building UPGMA tree...[/cyan]")
        root = build_upgma_tree(dist_matrix, ordered_names)

    if bootstrap_replicates and bootstrap_replicates > 0 and n >= 3:
        console.print(
            f"[cyan]Running bootstrap with {bootstrap_replicates} replicates...[/cyan]"
        )
        engine = AlignmentEngine(mode=distance_mode)
        aligned_pairs = engine.align_multiple(sequences, names)
        aligned_seqs = [pair[1] for pair in sorted(aligned_pairs, key=lambda p: names.index(p[0]))]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Bootstrap resampling...", total=bootstrap_replicates)

            def bs_progress(done, total):
                progress.update(task, completed=done, total=total)

            bs_values, orig_partitions = compute_bootstrap_support(
                aligned_seqs,
                names,
                method=method,
                n_replicates=bootstrap_replicates,
                mode=distance_mode,
                progress_callback=bs_progress,
            )
        assign_bootstrap_to_nodes(root, bs_values, orig_partitions)
        console.print("[green]Bootstrap support computed![/green]")

    console.print()
    console.print(f"[bold green]Phylogenetic tree ({method.upper()})[/bold green]")
    console.print()
    render_tree_ascii(root, console=console)

    if output_file:
        write_newick(root, output_file)
        console.print(f"[green]Newick tree saved to {output_file}[/green]")
    else:
        nwk = root.to_newick() + ";"
        console.print()
        console.print("[bold]Newick format:[/bold]")
        console.print(nwk)


@cli.command()
@click.option("-i", "--input", "input_file", required=True, help="Input FASTA/FASTQ file")
def stats(input_file):
    """Compute sequence statistics: GC content, length distribution, base composition."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    console.print(f"[green]Loaded {len(records)} sequences[/green]")

    result = compute_sequence_stats(records)
    render_stats(result, console=console)


def __make_record(name, seq):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    return SeqRecord(Seq(seq), id=name, description="")


if __name__ == "__main__":
    cli()
