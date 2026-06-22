from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from io_parser import NewickNode


def build_nj_tree(
    dist_matrix: np.ndarray, names: List[str], bootstrap: Optional[List[int]] = None
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
    current_names = list(range(n))
    nodes: dict[int, NewickNode] = {}
    for i in range(n):
        nodes[i] = NewickNode(name=names[i], distance=0.0)
    next_id = n

    active = list(range(n))

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

        if bootstrap and next_id < len(bootstrap) + n:
            new_node.bootstrap = bootstrap[next_id - n]

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
    dist_matrix: np.ndarray, names: List[str], bootstrap: Optional[List[int]] = None
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

        if bootstrap and next_id < len(bootstrap) + n:
            new_node.bootstrap = bootstrap[next_id - n]

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
