from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def _similarity_bg(value: float) -> str:
    if value > 0.85:
        return "color(22)"
    elif value >= 0.50:
        return "color(178)"
    else:
        return "color(240)"


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


def render_ascii_tree(
    root: NewickNode,
    console: Optional[Console] = None,
    max_width: int = 80,
) -> None:
    if console is None:
        console = Console()
    lines = _build_tree_lines(root, "")
    for line in lines:
        console.print(line)


def _build_tree_lines(
    node: NewickNode, prefix: str
) -> List[str]:
    lines = []
    if node.is_leaf:
        dist_str = f"({node.distance:.4f})" if node.distance > 0 else ""
        lines.append(f"{prefix}── {node.name} {dist_str}")
        return lines

    label = ""
    if node.bootstrap is not None:
        label = f" [{node.bootstrap:.0f}]"

    dist_str = f"({node.distance:.4f})" if node.distance > 0 else ""

    if len(node.children) == 1:
        lines.append(f"{prefix}──{label}{dist_str}")
        lines.extend(_build_tree_lines(node.children[0], prefix + "   "))
        return lines

    last_idx = len(node.children) - 1
    for idx, child in enumerate(node.children):
        if idx == 0 and len(node.children) > 1:
            if idx == last_idx:
                connector = "──"
                child_prefix = prefix + "   "
            else:
                connector = "┬─"
                child_prefix = prefix + "│  "
            if idx == last_idx:
                lines.append(f"{prefix}└{connector}{label}{dist_str}")
            else:
                lines.append(f"{prefix}┌{connector}{label}{dist_str}")
            lines.extend(_build_tree_lines(child, child_prefix))
        elif idx == last_idx:
            lines.extend(_build_tree_lines(child, prefix + "   "))
        else:
            lines.extend(_build_tree_lines(child, prefix + "│  "))

    return lines


def render_tree_ascii(
    root: NewickNode,
    console: Optional[Console] = None,
) -> None:
    if console is None:
        console = Console()
    tree_str = _render_node(root, "", True)
    console.print(tree_str)


def _render_node(node: NewickNode, prefix: str, is_last: bool) -> str:
    result = ""
    connector = "└──" if is_last else "├──"
    child_prefix = "    " if is_last else "│   "

    if node.is_leaf:
        dist_str = f":{node.distance:.4f}" if node.distance > 0 else ""
        result = prefix + connector + " " + node.name + dist_str + "\n"
    else:
        label = ""
        if node.bootstrap is not None:
            label = f" [{node.bootstrap:.0f}]"
        dist_str = f":{node.distance:.4f}" if node.distance > 0 else ""
        result = prefix + connector + label + dist_str + "\n"
        for i, child in enumerate(node.children):
            result += _render_node(
                child, prefix + child_prefix, i == len(node.children) - 1
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
