from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from aligner import AlignmentEngine
from io_parser import save_checkpoint, load_checkpoint


BATCH_SIZE = 100


class SimilarityMatrix:
    def __init__(self, names: List[str]):
        self.names = names
        self.n = len(names)
        self.matrix = np.zeros((self.n, self.n), dtype=np.float64)
        np.fill_diagonal(self.matrix, 1.0)

    def set(self, i: int, j: int, value: float) -> None:
        self.matrix[i][j] = value
        self.matrix[j][i] = value

    def get(self, i: int, j: int) -> float:
        return float(self.matrix[i][j])

    def to_distance_matrix(self) -> np.ndarray:
        return 1.0 - self.matrix


def compute_similarity_matrix(
    sequences: List[str],
    names: List[str],
    mode: str = "needle",
    checkpoint_dir: str | None = None,
    resume: bool = False,
    progress_callback=None,
) -> SimilarityMatrix:
    n = len(sequences)
    sim_mat = SimilarityMatrix(names)
    completed_pairs: set = set()

    if resume and checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
        if ckpt_path.exists():
            ck_names, ck_matrix, ck_pairs = load_checkpoint(ckpt_path)
            if ck_names == names:
                sim_mat.matrix = np.array(ck_matrix, dtype=np.float64)
                completed_pairs = ck_pairs

    engine = AlignmentEngine(mode=mode)
    total_pairs = n * (n - 1) // 2
    done_pairs = len(completed_pairs)

    if progress_callback:
        progress_callback(done_pairs, total_pairs)

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in completed_pairs:
                continue
            similarity = engine.align_pair_simple(sequences[i], sequences[j])
            sim_mat.set(i, j, similarity)
            completed_pairs.add((i, j))
            done_pairs += 1

            if (
                checkpoint_dir
                and done_pairs % BATCH_SIZE == 0
            ):
                ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
                save_checkpoint(
                    ckpt_path,
                    names,
                    sim_mat.matrix.tolist(),
                    completed_pairs,
                )

            if progress_callback and done_pairs % 10 == 0:
                progress_callback(done_pairs, total_pairs)

    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
        save_checkpoint(
            ckpt_path,
            names,
            sim_mat.matrix.tolist(),
            completed_pairs,
        )

    return sim_mat


def compute_distance_matrix(
    sequences: List[str],
    names: List[str],
    mode: str = "needle",
    checkpoint_dir: str | None = None,
    resume: bool = False,
    progress_callback=None,
) -> Tuple[np.ndarray, List[str]]:
    sim_mat = compute_similarity_matrix(
        sequences, names, mode, checkpoint_dir, resume, progress_callback
    )
    return sim_mat.to_distance_matrix(), sim_mat.names


def condense_matrix(dist_matrix: np.ndarray) -> List[float]:
    n = dist_matrix.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(float(dist_matrix[i, j]))
    return condensed
