from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


DISTANCE_MODELS = ["similarity", "p-distance", "jc69", "k2p"]


def _count_sites(aligned1: str, aligned2: str) -> Tuple[int, int, int, int]:
    """Count valid sites, transitions, transversions between two aligned sequences.

    Returns: (valid_sites, matches, transitions, transversions)
    """
    valid = 0
    matches = 0
    transitions = 0
    transversions = 0
    transition_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}

    for c1, c2 in zip(aligned1.upper(), aligned2.upper()):
        if c1 == "-" or c2 == "-":
            continue
        if c1 not in "ATGC" or c2 not in "ATGC":
            continue
        valid += 1
        if c1 == c2:
            matches += 1
        else:
            if (c1, c2) in transition_pairs:
                transitions += 1
            else:
                transversions += 1

    return valid, matches, transitions, transversions


def p_distance(aligned1: str, aligned2: str) -> float:
    """Proportion of different sites (p-distance)."""
    valid, matches, _tr, _tv = _count_sites(aligned1, aligned2)
    if valid == 0:
        return 0.0
    return 1.0 - matches / valid


def jukes_cantor_69(aligned1: str, aligned2: str) -> float:
    """Jukes-Cantor 1969 one-parameter distance."""
    p = p_distance(aligned1, aligned2)
    if p >= 0.75:
        return float("inf")
    try:
        d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
        return max(0.0, d)
    except ValueError:
        return float("inf")


def kimura_2p(aligned1: str, aligned2: str) -> float:
    """Kimura 2-parameter distance."""
    valid, matches, transitions, transversions = _count_sites(aligned1, aligned2)
    if valid == 0:
        return 0.0
    p = transitions / valid
    q = transversions / valid
    denom1 = 1.0 - 2.0 * p - q
    denom2 = 1.0 - 2.0 * q
    if denom1 <= 0 or denom2 <= 0:
        return float("inf")
    try:
        d = -0.5 * math.log(denom1) - 0.25 * math.log(denom2)
        return max(0.0, d)
    except ValueError:
        return float("inf")


def similarity_distance(aligned1: str, aligned2: str) -> float:
    """Distance based on fraction of matching positions (including gaps as non-match)."""
    if len(aligned1) == 0 and len(aligned2) == 0:
        return 0.0
    align_len = max(len(aligned1), len(aligned2))
    a1 = aligned1.ljust(align_len, "-")
    a2 = aligned2.ljust(align_len, "-")
    matches = 0
    for c1, c2 in zip(a1, a2):
        if c1 == "-" and c2 == "-":
            continue
        if c1 == c2:
            matches += 1
    return 1.0 - matches / align_len


DISTANCE_FUNCS = {
    "similarity": similarity_distance,
    "p-distance": p_distance,
    "jc69": jukes_cantor_69,
    "k2p": kimura_2p,
}


def compute_pairwise_distance(
    aligned1: str, aligned2: str, model: str = "p-distance"
) -> float:
    """Compute pairwise distance using the specified model."""
    if model not in DISTANCE_FUNCS:
        raise ValueError(f"Unknown distance model: {model}. Available: {list(DISTANCE_FUNCS.keys())}")
    return DISTANCE_FUNCS[model](aligned1, aligned2)


def compute_distance_matrix_from_aligned(
    aligned_sequences: List[str],
    names: List[str],
    model: str = "p-distance",
) -> Tuple[np.ndarray, List[str]]:
    """Compute distance matrix directly from pre-aligned sequences."""
    n = len(aligned_sequences)
    if len(names) != n:
        raise ValueError("Mismatch between number of sequences and names")

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    func = DISTANCE_FUNCS[model]

    for i in range(n):
        for j in range(i + 1, n):
            d = func(aligned_sequences[i], aligned_sequences[j])
            if d == float("inf"):
                d = 10.0
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix, names
