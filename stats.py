from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from Bio.SeqRecord import SeqRecord
from rich.console import Console
from rich.table import Table

from visualizer import render_bar_chart


def compute_gc_content(seq: str) -> float:
    seq = seq.upper()
    valid = sum(1 for c in seq if c in "ATGC")
    if valid == 0:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / valid


def compute_base_composition(seq: str) -> Dict[str, float]:
    seq = seq.upper()
    counter = Counter(c for c in seq if c in "ATGC")
    total = sum(counter.values())
    if total == 0:
        return {b: 0.0 for b in "ATGC"}
    return {b: counter.get(b, 0) / total for b in "ATGC"}


def compute_sequence_stats(
    records: List[SeqRecord],
) -> Dict:
    gc_contents = {}
    lengths = {}
    base_counts = Counter()
    per_seq_gc = {}

    for rec in records:
        seq = str(rec.seq)
        name = rec.id
        gc = compute_gc_content(seq)
        gc_contents[name] = gc
        per_seq_gc[name] = gc
        lengths[name] = len(seq)
        for c in seq.upper():
            if c in "ATGC":
                base_counts[c] += 1

    total_bases = sum(base_counts.values())
    overall_comp = {}
    if total_bases > 0:
        for b in "ATGC":
            overall_comp[b] = base_counts.get(b, 0) / total_bases
    else:
        overall_comp = {b: 0.0 for b in "ATGC"}

    length_values = list(lengths.values())
    mean_len = sum(length_values) / len(length_values) if length_values else 0
    min_len = min(length_values) if length_values else 0
    max_len = max(length_values) if length_values else 0

    gc_values = list(gc_contents.values())
    mean_gc = sum(gc_values) / len(gc_values) if gc_values else 0
    min_gc = min(gc_values) if gc_values else 0
    max_gc = max(gc_values) if gc_values else 0

    return {
        "gc_contents": gc_contents,
        "lengths": lengths,
        "overall_composition": overall_comp,
        "mean_length": mean_len,
        "min_length": min_len,
        "max_length": max_len,
        "mean_gc": mean_gc,
        "min_gc": min_gc,
        "max_gc": max_gc,
        "total_sequences": len(records),
    }


def render_stats(stats: Dict, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    table = Table(title="Sequence Statistics Summary", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Sequences", str(stats["total_sequences"]))
    table.add_row("Mean Length", f"{stats['mean_length']:.1f}")
    table.add_row("Min Length", str(stats["min_length"]))
    table.add_row("Max Length", str(stats["max_length"]))
    table.add_row("Mean GC%", f"{stats['mean_gc']:.2%}")
    table.add_row("Min GC%", f"{stats['min_gc']:.2%}")
    table.add_row("Max GC%", f"{stats['max_gc']:.2%}")
    console.print(table)

    comp = stats["overall_composition"]
    comp_data = {f"Base {b}": v * 100 for b, v in comp.items()}
    render_bar_chart(comp_data, "Overall Base Composition (%)", color="green", console=console)

    gc_contents = stats["gc_contents"]
    gc_data = {}
    for name, gc in sorted(gc_contents.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]:
        gc_data[name[:15]] = gc * 100
    render_bar_chart(gc_data, "GC Content per Sequence (%)", color="yellow", console=console)

    lengths = stats["lengths"]
    len_data = {}
    for name, length in sorted(lengths.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]:
        len_data[name[:15]] = float(length)
    render_bar_chart(len_data, "Sequence Lengths (top 20)", color="cyan", console=console)
