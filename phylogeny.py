from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from io_parser import NewickNode
from matrix import compute_distance_matrix


def build_nj_tree(
    dist_matrix: np.ndarray,
    names: List[str],
    bootstrap: Optional[List[float]] = None,
) -> NewickNode:
    n = dist_matrix.shape[0]
    if n == 0:
        raise ValueError("Empty distance matrix")
    if n == 1:
        return NewickNode(name=names[0], distance=0.0)
    if n == 2:
        d = float(dist_matrix[0, 1])
        root = NewickNode(distance=0.0)
        root.children = [
            NewickNode(name=names[0], distance=d / 2),
            NewickNode(name=names[1], distance=d / 2),
        ]
        return root

    dm = dist_matrix.copy().astype(float)
    nodes: dict[int, NewickNode] = {}
    for i in range(n):
        nodes[i] = NewickNode(name=names[i], distance=0.0)
    next_id = n

    active = list(range(n))
    bootstrap_idx = 0

    while len(active) > 2:
        r = len(active)
        q = np.full((r, r), np.inf)
        for ai in range(r):
            for aj in range(ai + 1, r):
                i, j = active[ai], active[aj]
                sum_i = np.sum(dm[i, active])
                sum_j = np.sum(dm[j, active])
                q[ai, aj] = (r - 2) * dm[i, j] - sum_i - sum_j
                q[aj, ai] = q[ai, aj]

        min_val = np.inf
        min_ai, min_aj = 0, 1
        for ai in range(r):
            for aj in range(ai + 1, r):
                if q[ai, aj] < min_val:
                    min_val = q[ai, aj]
                    min_ai, min_aj = ai, aj

        i, j = active[min_ai], active[min_aj]
        sum_i = np.sum(dm[i, active])
        sum_j = np.sum(dm[j, active])
        d_ij = dm[i, j]

        if r - 2 != 0:
            d_iu = max(0.0, 0.5 * d_ij + (sum_i - sum_j) / (2 * (r - 2)))
            d_ju = max(0.0, d_ij - d_iu)
        else:
            d_iu = d_ij / 2
            d_ju = d_ij / 2

        new_node = NewickNode(distance=0.0)
        nodes[i].distance = d_iu
        nodes[j].distance = d_ju
        new_node.children = [nodes[i], nodes[j]]

        if bootstrap is not None and bootstrap_idx < len(bootstrap):
            new_node.bootstrap = bootstrap[bootstrap_idx]
            bootstrap_idx += 1

        new_id = next_id
        nodes[new_id] = new_node
        next_id += 1

        new_row = np.full(dm.shape[0], 0.0)
        for k in active:
            if k == i or k == j:
                continue
            d_uk = 0.5 * (dm[i, k] + dm[j, k] - d_ij)
            new_row[k] = max(0.0, d_uk)
            dm[k, i] = new_row[k]
            dm[i, k] = new_row[k]

        dm = np.vstack([dm, new_row.reshape(1, -1)])
        new_col = np.append(new_row, 0.0)
        dm = np.hstack([dm, new_col.reshape(-1, 1)])

        active.remove(i)
        active.remove(j)
        active.append(new_id)

    if len(active) == 2:
        i, j = active[0], active[1]
        d = float(dm[i, j])
        nodes[i].distance = d / 2
        nodes[j].distance = d / 2
        root = NewickNode(distance=0.0)
        root.children = [nodes[i], nodes[j]]
        return root
    else:
        return nodes[active[0]]


def build_upgma_tree(
    dist_matrix: np.ndarray,
    names: List[str],
    bootstrap: Optional[List[float]] = None,
) -> NewickNode:
    n = dist_matrix.shape[0]
    if n == 0:
        raise ValueError("Empty distance matrix")
    if n == 1:
        return NewickNode(name=names[0], distance=0.0)

    dm = dist_matrix.copy().astype(float)
    nodes: dict[int, NewickNode] = {}
    cluster_sizes: dict[int, int] = {}
    for i in range(n):
        nodes[i] = NewickNode(name=names[i], distance=0.0)
        cluster_sizes[i] = 1
    next_id = n

    active = list(range(n))
    bootstrap_idx = 0

    while len(active) > 1:
        min_val = np.inf
        min_i, min_j = active[0], active[1]
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                i, j = active[ai], active[aj]
                if dm[i, j] < min_val:
                    min_val = dm[i, j]
                    min_i, min_j = i, j

        height = min_val / 2
        new_node = NewickNode(distance=0.0)
        nodes[min_i].distance = max(0.0, height - _node_height(nodes[min_i]))
        nodes[min_j].distance = max(0.0, height - _node_height(nodes[min_j]))
        new_node.children = [nodes[min_i], nodes[min_j]]

        if bootstrap is not None and bootstrap_idx < len(bootstrap):
            new_node.bootstrap = bootstrap[bootstrap_idx]
            bootstrap_idx += 1

        new_id = next_id
        nodes[new_id] = new_node
        next_id += 1

        size_i = cluster_sizes[min_i]
        size_j = cluster_sizes[min_j]
        cluster_sizes[new_id] = size_i + size_j

        new_row = np.full(dm.shape[0], 0.0)
        for k in active:
            if k == min_i or k == min_j:
                continue
            d_uk = (size_i * dm[min_i, k] + size_j * dm[min_j, k]) / (
                size_i + size_j
            )
            new_row[k] = d_uk
            dm[k, min_i] = d_uk
            dm[min_i, k] = d_uk

        dm = np.vstack([dm, new_row.reshape(1, -1)])
        new_col = np.append(new_row, 0.0)
        dm = np.hstack([dm, new_col.reshape(-1, 1)])

        active.remove(min_i)
        active.remove(min_j)
        active.append(new_id)

    return nodes[active[0]]


def _node_height(node: NewickNode) -> float:
    if node.is_leaf:
        return 0.0
    return max(
        child.distance + _node_height(child) for child in node.children
    )


def collect_leaf_names(node: NewickNode) -> List[str]:
    if node.is_leaf:
        return [node.name]
    result = []
    for child in node.children:
        result.extend(collect_leaf_names(child))
    return result


def get_internal_partitions(root: NewickNode) -> List[Tuple[frozenset, frozenset]]:
    partitions: List[Tuple[frozenset, frozenset]] = []

    def _walk(node: NewickNode, all_leaves: frozenset) -> frozenset:
        if node.is_leaf:
            return frozenset([node.name])
        child_leaves = []
        for child in node.children:
            child_leaves.append(_walk(child, all_leaves))
        combined = frozenset().union(*child_leaves)
        if len(child_leaves) >= 2 and len(combined) < len(all_leaves) and len(combined) > 1:
            complement = all_leaves - combined
            if len(complement) > 1:
                smaller = min(combined, complement, key=lambda x: (len(x), sorted(x)))
                larger = all_leaves - smaller
                partitions.append((frozenset(smaller), frozenset(larger)))
        return combined

    all_leaves = frozenset(collect_leaf_names(root))
    _walk(root, all_leaves)
    return partitions


def get_internal_nodes_with_leaves(root: NewickNode) -> List[Tuple[NewickNode, frozenset]]:
    result: List[Tuple[NewickNode, frozenset]] = []

    def _walk(node: NewickNode) -> frozenset:
        if node.is_leaf:
            return frozenset([node.name])
        child_leaves = []
        for child in node.children:
            child_leaves.append(_walk(child))
        combined = frozenset().union(*child_leaves)
        if len(child_leaves) >= 2 and len(combined) > 1 and not (node is root and len(result) == 0):
            pass
        if not node.is_leaf and len(combined) > 1:
            result.append((node, combined))
        return combined

    _walk(root)
    return result


def assign_bootstrap_to_nodes(
    root: NewickNode, bootstrap_values: List[float], orig_partitions: List[frozenset]
) -> None:
    all_leaves = frozenset(collect_leaf_names(root))
    internal = get_internal_nodes_with_leaves(root)

    for node, leaves in internal:
        if len(leaves) <= 1 or len(leaves) >= len(all_leaves):
            continue
        smaller = min(leaves, all_leaves - leaves, key=lambda x: (len(x), sorted(x)))
        partition_key = frozenset(smaller)
        for pi, p in enumerate(orig_partitions):
            if p == partition_key:
                node.bootstrap = bootstrap_values[pi]
                break


def compute_bootstrap_support(
    aligned_sequences: List[str],
    names: List[str],
    method: str = "nj",
    n_replicates: int = 100,
    mode: str = "needle",
    distance_model: Optional[str] = None,
    random_seed: Optional[int] = 42,
    progress_callback=None,
) -> Tuple[List[float], List[frozenset]]:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    n = len(names)
    if n < 3:
        return [], []

    seq_len = len(aligned_sequences[0])
    if seq_len == 0:
        return [], []

    if distance_model and distance_model != "similarity":
        from distance import compute_distance_matrix_from_aligned
        orig_dist, _ = compute_distance_matrix_from_aligned(
            aligned_sequences, names, model=distance_model
        )
    else:
        orig_dist, _ = compute_distance_matrix(aligned_sequences, names, mode=mode)
    if method == "nj":
        orig_tree = build_nj_tree(orig_dist, names)
    else:
        orig_tree = build_upgma_tree(orig_dist, names)

    all_leaves = frozenset(names)

    orig_partitions: List[frozenset] = []
    internal_nodes_info = get_internal_nodes_with_leaves(orig_tree)
    for node, leaves in internal_nodes_info:
        if len(leaves) <= 1 or len(leaves) >= len(all_leaves):
            continue
        smaller = min(leaves, all_leaves - leaves, key=lambda x: (len(x), sorted(x)))
        orig_partitions.append(frozenset(smaller))

    partition_counts = [0] * len(orig_partitions)

    for rep in range(n_replicates):
        indices = sorted(random.choices(range(seq_len), k=seq_len))
        resampled = ["".join(seq[i] for i in indices) for seq in aligned_sequences]

        try:
            if distance_model and distance_model != "similarity":
                rep_dist, _ = compute_distance_matrix_from_aligned(
                    resampled, names, model=distance_model
                )
            else:
                rep_dist, _ = compute_distance_matrix(resampled, names, mode=mode)
            if method == "nj":
                rep_tree = build_nj_tree(rep_dist, names)
            else:
                rep_tree = build_upgma_tree(rep_dist, names)

            rep_partitions_raw = get_internal_nodes_with_leaves(rep_tree)
            rep_partitions = set()
            for _node, leaves in rep_partitions_raw:
                if len(leaves) <= 1 or len(leaves) >= len(all_leaves):
                    continue
                smaller = min(leaves, all_leaves - leaves, key=lambda x: (len(x), sorted(x)))
                rep_partitions.add(frozenset(smaller))

            for pi, p in enumerate(orig_partitions):
                if p in rep_partitions:
                    partition_counts[pi] += 1
        except Exception:
            continue

        if progress_callback and (rep + 1) % 5 == 0:
            progress_callback(rep + 1, n_replicates)

    bootstrap_values = [c * 100.0 / n_replicates for c in partition_counts]
    return bootstrap_values, orig_partitions
