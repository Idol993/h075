from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from io_parser import NewickNode


def _similarity_color(value: float) -> str:
    if value > 0.85:
        return "green"
    elif value >= 0.50:
        return "yellow"
    else:
        return "dim"


def render_heatmap(
    sim_matrix: np.ndarray,
    names: List[str],
    console: Optional[Console] = None,
) -> None:
    if console is None:
        console = Console()
    n = len(names)
    max_name_len = max(len(nm) for nm in names) if names else 5
    truncated = [nm[:max_name_len] for nm in names]

    table = Table(title="Similarity Matrix Heatmap", show_lines=False)
    table.add_column("", style="bold", min_width=max_name_len)
    for name in truncated:
        table.add_column(name, justify="center", min_width=6)

    for i in range(n):
        row_cells = []
        for j in range(n):
            val = sim_matrix[i, j]
            color = _similarity_color(val)
            cell = Text(f"{val:.2f}", style=color)
            row_cells.append(cell)
        table.add_row(truncated[i], *row_cells)

    console.print(table)
    console.print()
    console.print("[green]■[/green] >85%  [yellow]■[/yellow] 50-85%  [dim]■[/dim] <50%")


def _tree_max_depth(node: NewickNode) -> float:
    if node.is_leaf:
        return node.distance
    return node.distance + max(
        _tree_max_depth(child) for child in node.children
    )


def render_tree_ascii(
    root: NewickNode,
    console: Optional[Console] = None,
    max_branch_chars: int = 40,
) -> None:
    if console is None:
        console = Console()
    max_depth = _tree_max_depth(root)
    if max_depth <= 0:
        max_depth = 1.0
    scale = max_branch_chars / max_depth

    lines = _render_scaled_node(root, "", True, scale, 0.0)
    console.print(lines.rstrip())


def _render_scaled_node(
    node: NewickNode,
    prefix: str,
    is_last: bool,
    scale: float,
    cum_dist: float,
) -> str:
    branch_chars = max(1, int(round(node.distance * scale)))
    branch_line = "─" * branch_chars
    result = ""

    connector = "└" if is_last else "├"
    child_prefix_cont = prefix + (" " * (branch_chars + 3)) if is_last else prefix + "│" + (" " * (branch_chars + 2))

    if node.is_leaf:
        label = node.name
        if node.bootstrap is not None:
            label += f" [{node.bootstrap:.0f}]"
        label += f" ({node.distance:.4f})"
        result = prefix + connector + branch_line + " " + label + "\n"
    else:
        node_label = ""
        if node.bootstrap is not None:
            node_label = f" [{node.bootstrap:.0f}]"
        if node.distance > 0:
            node_label += f" ({node.distance:.4f})"
        result = prefix + connector + branch_line + node_label + "\n"

        for i, child in enumerate(node.children):
            child_is_last = (i == len(node.children) - 1)
            result += _render_scaled_node(
                child,
                child_prefix_cont if not node.is_leaf else prefix + (" " * (branch_chars + 3)),
                child_is_last,
                scale,
                cum_dist + node.distance,
            )
    return result


def render_bar_chart(
    data: Dict[str, float],
    title: str,
    width: int = 40,
    console: Optional[Console] = None,
    color: str = "cyan",
) -> None:
    if console is None:
        console = Console()
    if not data:
        return
    max_val = max(data.values()) if data.values() else 1.0
    if max_val == 0:
        max_val = 1.0

    table = Table(title=title, show_lines=False)
    table.add_column("Label", style="bold", min_width=12)
    table.add_column("Value", justify="right", min_width=8)
    table.add_column("Bar", min_width=width + 2)

    for label, value in data.items():
        bar_len = int((value / max_val) * width)
        bar = "█" * bar_len + "░" * (width - bar_len)
        table.add_row(label, f"{value:.2f}", f"[{color}]{bar}[/{color}]")

    console.print(table)
