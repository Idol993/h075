from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

from io_parser import read_sequences, write_fasta, write_newick, NewickNode, load_checkpoint
from aligner import AlignmentEngine
from matrix import (
    compute_similarity_matrix,
    compute_distance_matrix,
    compute_distance_matrix_from_aligned,
    compute_similarity_from_aligned,
    is_alignment_input,
    pair_align_all,
)
from phylogeny import (
    build_nj_tree,
    build_upgma_tree,
    collect_leaf_names,
    compute_bootstrap_support,
    assign_bootstrap_to_nodes,
    get_internal_nodes_with_leaves,
)
from visualizer import render_heatmap, render_tree_ascii, render_bar_chart
from stats import compute_sequence_stats, render_stats
from distance import DISTANCE_MODELS


console = Console()


DISTANCE_MODEL_HELP = (
    "Distance model: similarity, p-distance, jc69 (Jukes-Cantor), k2p (Kimura 2-parameter)"
)


@click.group()
@click.version_option("1.1.0")
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
@click.option("--shard-size", type=int, default=100, help="Shard size for batch processing (default 100)")
def align(input_file, mode, output_file, checkpoint_dir, resume, shard_size):
    """Pairwise align sequences and generate similarity matrix heatmap."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    names = [r.id for r in records]
    sequences = [str(r.seq) for r in records]
    n = len(records)
    console.print(f"[green]Loaded {n} sequences[/green]")

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    total_shards = (n + shard_size - 1) // shard_size
    if n > 2000:
        console.print(
            f"[yellow]Large dataset ({n} sequences). "
            f"Batch processing enabled ({total_shards} shards × {shard_size} seqs).[/yellow]"
        )

    start_shard = 0
    if resume and checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
        if ckpt_path.exists():
            try:
                ck_names, _ck_mat, ck_pairs = load_checkpoint(ckpt_path)
                if ck_names == names:
                    total_pairs_all = n * (n - 1) // 2
                    done_pairs_all = len(ck_pairs)
                    start_shard = _estimate_shard_from_pairs(ck_pairs, n, shard_size)
                    pct = done_pairs_all / total_pairs_all * 100 if total_pairs_all > 0 else 0
                    console.print(
                        f"[cyan]Resuming from shard {start_shard}/{total_shards} "
                        f"({done_pairs_all}/{total_pairs_all} pairs, {pct:.1f}%)[/cyan]"
                    )
            except Exception:
                pass

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
            shard_size=shard_size,
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


@cli.command(name="matrix")
@click.option("-i", "--input", "input_file", required=True, help="Input FASTA file (aligned or unaligned)")
@click.option(
    "-t",
    "--type",
    "matrix_type",
    type=click.Choice(["distance", "similarity"]),
    default="distance",
    help="Matrix type to export",
)
@click.option(
    "-d",
    "--distance-model",
    type=click.Choice(DISTANCE_MODELS),
    default="p-distance",
    help=DISTANCE_MODEL_HELP,
)
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["csv", "tsv"]),
    default="csv",
    help="Output format",
)
@click.option("-o", "--output", "output_file", required=True, help="Output matrix file")
@click.option("--align-mode", type=click.Choice(["needle", "water"]), default="needle",
              help="Alignment mode if input is unaligned")
@click.option("--checkpoint-dir", default=None, help="Directory for checkpoint files")
@click.option("--resume", is_flag=True, default=False, help="Resume from checkpoint")
def matrix_cmd(input_file, matrix_type, distance_model, fmt, output_file, align_mode, checkpoint_dir, resume):
    """Export distance/similarity matrix as CSV or TSV."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    names = [r.id for r in records]
    sequences = [str(r.seq) for r in records]
    n = len(records)
    console.print(f"[green]Loaded {n} sequences[/green]")

    aligned = is_alignment_input(sequences)

    if aligned:
        has_gaps = any("-" in s for s in sequences)
        gap_str = ", with gaps" if has_gaps else ", no gaps"
        console.print(
            f"[cyan]Detected equal-length input ({len(sequences[0])} bp{gap_str}). "
            f"Treating as pre-aligned; using column positions directly.[/cyan]"
        )
        if matrix_type == "distance":
            console.print(f"[cyan]Computing distance matrix with model: {distance_model}[/cyan]")
            mat, ordered_names = compute_distance_matrix_from_aligned(
                sequences, names, model=distance_model
            )
        else:
            console.print("[cyan]Computing similarity matrix from aligned sequences[/cyan]")
            mat, ordered_names = compute_similarity_from_aligned(sequences, names)
    else:
        console.print(f"[yellow]Unaligned input. Running pairwise {align_mode} alignment first...[/yellow]")
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Aligning pairs...", total=n * (n - 1) // 2)

            def on_progress(done, total):
                progress.update(task, completed=done, total=total)

            aligned_pairs, sim_mat = pair_align_all(
                sequences, names, mode=align_mode,
                checkpoint_dir=checkpoint_dir, resume=resume,
                progress_callback=on_progress,
            )

        aligned_seqs = [seq for _name, seq in sorted(aligned_pairs, key=lambda p: names.index(p[0]))]

        if matrix_type == "distance":
            if distance_model == "similarity":
                mat = sim_mat.to_distance_matrix()
                ordered_names = sim_mat.names
            else:
                console.print(f"[cyan]Computing distance matrix with model: {distance_model}[/cyan]")
                mat, ordered_names = compute_distance_matrix_from_aligned(
                    aligned_seqs, names, model=distance_model
                )
        else:
            mat = sim_mat.matrix
            ordered_names = sim_mat.names

    delimiter = "," if fmt == "csv" else "\t"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# matrix_type={matrix_type}\n")
        fh.write(f"# distance_model={distance_model}\n")
        if aligned:
            fh.write("# input=aligned (equal-length sequences)\n")
        else:
            fh.write(f"# input=unaligned; pairwise_{align_mode}_alignment_used\n")
        writer = csv.writer(fh, delimiter=delimiter)
        header = [""] + ordered_names
        writer.writerow(header)
        for i, name in enumerate(ordered_names):
            row = [name] + [f"{mat[i, j]:.6f}" for j in range(len(ordered_names))]
            writer.writerow(row)

    console.print(f"[green]Matrix saved to {output_file} ({fmt.upper()}, {matrix_type})[/green]")
    if matrix_type == "distance":
        console.print(f"[green]  Distance model: {distance_model}[/green]")

    table = Table(title=f"{matrix_type.capitalize()} Matrix (top-left preview)")
    show_n = min(5, len(ordered_names))
    table.add_column("", style="bold")
    for nm in ordered_names[:show_n]:
        table.add_column(nm[:12], justify="right")
    for i in range(show_n):
        row = [ordered_names[i][:12]]
        for j in range(show_n):
            row.append(f"{mat[i, j]:.4f}")
        table.add_row(*row)
    console.print(table)


@cli.command()
@click.option("-i", "--input", "input_file", required=True, help="Input FASTA (aligned or unaligned)")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["nj", "upgma"]),
    default="nj",
    help="Tree construction method: nj (Neighbor-Joining) or upgma",
)
@click.option("-o", "--output", "output_file", default=None, help="Output Newick file")
@click.option(
    "--distance-model",
    type=click.Choice(DISTANCE_MODELS),
    default="p-distance",
    help=DISTANCE_MODEL_HELP,
)
@click.option("--align-mode", type=click.Choice(["needle", "water"]), default="needle",
              help="Alignment mode if input is unaligned")
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
@click.option("--report", "report_file", default=None, help="Output tree report (CSV) with internal branches")
def tree(input_file, method, output_file, distance_model, align_mode,
         bootstrap_replicates, checkpoint_dir, resume, report_file):
    """Build phylogenetic tree (NJ/UPGMA) with bootstrap and render ASCII preview."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    names = [r.id for r in records]
    sequences = [str(r.seq) for r in records]
    n = len(records)
    console.print(f"[green]Loaded {n} sequences[/green]")

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    pre_aligned = is_alignment_input(sequences)

    if pre_aligned:
        has_gaps = any("-" in s for s in sequences)
        gap_str = ", with gaps" if has_gaps else ", no gaps"
        console.print(
            f"[cyan]Detected equal-length input ({len(sequences[0])} bp{gap_str}). "
            f"Treating as pre-aligned; using column positions directly.[/cyan]"
        )
        aligned_seqs = list(sequences)
        console.print(f"[cyan]Computing distance matrix (model: {distance_model})...[/cyan]")
        dist_matrix, ordered_names = compute_distance_matrix_from_aligned(
            aligned_seqs, names, model=distance_model
        )
    else:
        console.print(f"[yellow]Unaligned input. Running pairwise {align_mode} alignment...[/yellow]")
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

            if distance_model == "similarity":
                dist_matrix, ordered_names = compute_distance_matrix(
                    sequences, names, mode=align_mode,
                    checkpoint_dir=checkpoint_dir, resume=resume,
                    progress_callback=on_progress,
                )
                aligned_seqs = None
            else:
                engine = AlignmentEngine(mode=align_mode)
                aligned_pairs = engine.align_multiple(sequences, names)
                aligned_seqs = [seq for _nm, seq in sorted(aligned_pairs, key=lambda p: names.index(p[0]))]
                dist_matrix, ordered_names = compute_distance_matrix_from_aligned(
                    aligned_seqs, names, model=distance_model
                )

    console.print()
    console.print(f"[bold green]Distance matrix computed (model: {distance_model})![/bold green]")

    if method == "nj":
        console.print(f"[cyan]Building Neighbor-Joining tree (model: {distance_model})...[/cyan]")
        root = build_nj_tree(dist_matrix, ordered_names)
    else:
        console.print(f"[cyan]Building UPGMA tree (model: {distance_model})...[/cyan]")
        root = build_upgma_tree(dist_matrix, ordered_names)

    if bootstrap_replicates and bootstrap_replicates > 0 and n >= 3:
        if pre_aligned:
            aligned_for_bs = sequences
        elif aligned_seqs is not None:
            aligned_for_bs = aligned_seqs
        else:
            engine = AlignmentEngine(mode=align_mode)
            aligned_pairs = engine.align_multiple(sequences, names)
            aligned_for_bs = [seq for _nm, seq in sorted(aligned_pairs, key=lambda p: names.index(p[0]))]

        console.print(
            f"[cyan]Running bootstrap with {bootstrap_replicates} replicates...[/cyan]"
        )

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
                aligned_for_bs,
                names,
                method=method,
                n_replicates=bootstrap_replicates,
                mode=align_mode,
                distance_model=distance_model,
                progress_callback=bs_progress,
            )
        assign_bootstrap_to_nodes(root, bs_values, orig_partitions)
        console.print("[green]Bootstrap support computed![/green]")

    console.print()
    console.print(
        f"[bold green]Phylogenetic tree ({method.upper()}, {distance_model})[/bold green]"
    )
    console.print()
    render_tree_ascii(root, console=console)

    if output_file:
        _write_newick_with_metadata(root, output_file, method, distance_model, bootstrap_replicates)
        console.print(f"[green]Newick tree saved to {output_file}[/green]")
    else:
        nwk = root.to_newick() + ";"
        console.print()
        console.print("[bold]Newick format:[/bold]")
        console.print(nwk)

    if report_file:
        _write_tree_report(
            root, report_file,
            bootstrap_enabled=(bootstrap_replicates > 0),
            tree_method=method,
            distance_model=distance_model,
            bootstrap_replicates=bootstrap_replicates,
        )
        console.print(f"[green]Tree report saved to {report_file}[/green]")


def _write_newick_with_metadata(
    root: NewickNode, output_file: str,
    tree_method: str, distance_model: str, bootstrap_replicates: int,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(f"# tree_method={tree_method}\n")
        fh.write(f"# distance_model={distance_model}\n")
        if bootstrap_replicates > 0:
            fh.write(f"# bootstrap_replicates={bootstrap_replicates}\n")
        fh.write(root.to_newick() + ";\n")


def _write_tree_report(
    root: NewickNode, report_file: str, bootstrap_enabled: bool,
    tree_method: str = "", distance_model: str = "", bootstrap_replicates: int = 0,
) -> None:
    all_leaves = frozenset(collect_leaf_names(root))
    internal_info = get_internal_nodes_with_leaves(root)

    output_path = Path(report_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (node, leaves) in enumerate(internal_info):
        if len(leaves) <= 1 or len(leaves) >= len(all_leaves):
            continue
        leaf_list = sorted(leaves)
        branch_length = node.distance
        row = {
            "node_id": f"Node_{idx + 1}",
            "branch_length": f"{branch_length:.6f}",
            "num_leaves": len(leaf_list),
            "leaves": ",".join(leaf_list),
        }
        if bootstrap_enabled:
            row["bootstrap"] = f"{node.bootstrap:.1f}" if node.bootstrap is not None else ""
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        if tree_method:
            fh.write(f"# tree_method={tree_method}\n")
        if distance_model:
            fh.write(f"# distance_model={distance_model}\n")
        if bootstrap_replicates > 0:
            fh.write(f"# bootstrap_replicates={bootstrap_replicates}\n")
        if not rows:
            fh.write("# No internal branches (only 2 sequences or polytomy at root)\n")
            fh.write("# Sequences: " + ",".join(sorted(all_leaves)) + "\n")

        fieldnames = ["node_id", "branch_length", "num_leaves", "leaves"]
        if bootstrap_enabled:
            fieldnames.insert(2, "bootstrap")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@cli.command()
@click.option("-i", "--input", "input_file", required=True, help="Input FASTA/FASTQ file")
def stats(input_file):
    """Compute sequence statistics: GC content, length distribution, base composition."""
    console.print(f"[bold cyan]Loading sequences from {input_file}...[/bold cyan]")
    records = read_sequences(input_file)
    console.print(f"[green]Loaded {len(records)} sequences[/green]")

    result = compute_sequence_stats(records)
    render_stats(result, console=console)


def _estimate_shard_from_pairs(completed_pairs: set, n: int, shard_size: int) -> int:
    if not completed_pairs:
        return 0
    max_i = max(i for i, _j in completed_pairs)
    return max_i // shard_size


def __make_record(name, seq):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    return SeqRecord(Seq(seq), id=name, description="")


if __name__ == "__main__":
    cli()
