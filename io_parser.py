from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def read_sequences(filepath: str | Path) -> List[SeqRecord]:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix in (".fasta", ".fa", ".fna", ".fas"):
        fmt = "fasta"
    elif suffix in (".fastq", ".fq"):
        fmt = "fastq"
    else:
        fmt = _detect_format(filepath)
    records = list(SeqIO.parse(str(filepath), fmt))
    if not records:
        raise ValueError(f"No sequences found in {filepath}")
    return records


def _detect_format(filepath: Path) -> str:
    with open(filepath) as fh:
        first_char = fh.read(1)
    return "fastq" if first_char == "@" else "fasta"


def write_fasta(records: List[SeqRecord], filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(filepath), "fasta")


def write_aligned_fasta(
    aligned_pairs: List[Tuple[str, str]], filepath: str | Path
) -> None:
    records = [
        SeqRecord(Seq(aligned_seq), id=name, description="")
        for name, aligned_seq in aligned_pairs
    ]
    write_fasta(records, filepath)


class NewickNode:
    def __init__(
        self,
        name: str = "",
        distance: float = 0.0,
        children: List[NewickNode] | None = None,
        bootstrap: float | None = None,
    ):
        self.name = name
        self.distance = distance
        self.children = children if children is not None else []
        self.bootstrap = bootstrap

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_newick(self) -> str:
        if self.is_leaf:
            return f"{self.name}:{self.distance:.6f}"
        child_strs = ",".join(c.to_newick() for c in self.children)
        label = ""
        if self.bootstrap is not None:
            label = f"{self.bootstrap:.0f}"
        return f"({child_strs}){label}:{self.distance:.6f}"

    @staticmethod
    def from_newick(s: str) -> NewickNode:
        s = s.strip().rstrip(";").strip()
        node, _ = _parse_newick(s, 0)
        return node


def _parse_newick(s: str, pos: int) -> Tuple[NewickNode, int]:
    node = NewickNode()
    if pos < len(s) and s[pos] == "(":
        pos += 1
        while True:
            child, pos = _parse_newick(s, pos)
            node.children.append(child)
            if pos < len(s) and s[pos] == ",":
                pos += 1
            else:
                break
        if pos < len(s) and s[pos] == ")":
            pos += 1
    name_end = pos
    while name_end < len(s) and s[name_end] not in ":,);":
        name_end += 1
    label = s[pos:name_end].strip()
    if label:
        try:
            node.bootstrap = float(label)
        except ValueError:
            node.name = label
    pos = name_end
    if pos < len(s) and s[pos] == ":":
        pos += 1
        dist_end = pos
        while dist_end < len(s) and s[dist_end] not in ",);":
            dist_end += 1
        node.distance = float(s[pos:dist_end])
        pos = dist_end
    return node, pos


def write_newick(root: NewickNode, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as fh:
        fh.write(root.to_newick() + ";\n")


def read_newick(filepath: str | Path) -> NewickNode:
    filepath = Path(filepath)
    with open(filepath) as fh:
        content = fh.read().strip()
    return NewickNode.from_newick(content)


def save_checkpoint(
    filepath: str | Path,
    names: List[str],
    matrix: List[List[float]],
    completed_pairs: set,
) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "names": names,
        "matrix": matrix,
        "completed_pairs": [list(p) for p in completed_pairs],
    }
    with open(filepath, "w") as fh:
        json.dump(data, fh)


def load_checkpoint(
    filepath: str | Path,
) -> Tuple[List[str], List[List[float]], set]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    with open(filepath) as fh:
        data = json.load(fh)
    names = data["names"]
    matrix = data["matrix"]
    completed_pairs = {tuple(p) for p in data["completed_pairs"]}
    return names, matrix, completed_pairs
