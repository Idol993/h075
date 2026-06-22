from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from aligner import AlignmentEngine
from distance import (
    DISTANCE_FUNCS,
    DISTANCE_MODELS,
    compute_distance_matrix_from_aligned,
)
from io_parser import save_checkpoint, load_checkpoint


CHECKPOINT_PAIRS_BATCH = 100
SHARD_SIZE = 100


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
    shard_size: int = SHARD_SIZE,
) -> SimilarityMatrix:
    n = len(sequences)
    sim_mat = SimilarityMatrix(names)
    completed_pairs: set = set()
    last_shard = 0

    if resume and checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
        if ckpt_path.exists():
            ck_names, ck_matrix, ck_pairs = load_checkpoint(ckpt_path)
            if ck_names == names:
                sim_mat.matrix = np.array(ck_matrix, dtype=np.float64)
                completed_pairs = ck_pairs
                if progress_callback:
                    progress_callback(len(completed_pairs), n * (n - 1) // 2)
                last_shard = _find_shard_from_pairs(completed_pairs, n, shard_size)

    engine = AlignmentEngine(mode=mode)
    total_pairs = n * (n - 1) // 2
    done_pairs = len(completed_pairs)

    if n > 2000 and progress_callback:
        _report_shard(last_shard, n, shard_size)

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in completed_pairs:
                continue
            similarity = engine.align_pair_simple(sequences[i], sequences[j])
            sim_mat.set(i, j, similarity)
            completed_pairs.add((i, j))
            done_pairs += 1

            if checkpoint_dir and done_pairs % CHECKPOINT_PAIRS_BATCH == 0:
                ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
                save_checkpoint(
                    ckpt_path,
                    names,
                    sim_mat.matrix.tolist(),
                    completed_pairs,
                )

            if progress_callback and done_pairs % 10 == 0:
                progress_callback(done_pairs, total_pairs)

        if checkpoint_dir and (i + 1) % shard_size == 0:
            ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
            save_checkpoint(
                ckpt_path,
                names,
                sim_mat.matrix.tolist(),
                completed_pairs,
            )
            if progress_callback:
                _report_shard((i + 1) // shard_size, n, shard_size)

    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / "checkpoint.json"
        save_checkpoint(
            ckpt_path,
            names,
            sim_mat.matrix.tolist(),
            completed_pairs,
        )

    return sim_mat


def _find_shard_from_pairs(completed_pairs: set, n: int, shard_size: int) -> int:
    if not completed_pairs:
        return 0
    max_i = max(i for i, j in completed_pairs)
    return max_i // shard_size


def _report_shard(shard_idx: int, total_seqs: int, shard_size: int) -> None:
    total_shards = (total_seqs + shard_size - 1) // shard_size
    print(f"[shard {shard_idx}/{total_shards}] completed")


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


def compute_distance_matrix_from_aligned_safe(
    aligned_sequences: List[str],
    names: List[str],
    model: str = "p-distance",
) -> Tuple[np.ndarray, List[str]]:
    return compute_distance_matrix_from_aligned(aligned_sequences, names, model)


def compute_similarity_from_aligned(
    aligned_sequences: List[str],
    names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    n = len(aligned_sequences)
    sim_matrix = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a1 = aligned_sequences[i]
            a2 = aligned_sequences[j]
            align_len = max(len(a1), len(a2))
            matches = 0
            aa = a1.ljust(align_len, "-")
            bb = a2.ljust(align_len, "-")
            for c1, c2 in zip(aa, bb):
                if c1 == "-" and c2 == "-":
                    continue
                if c1 == c2:
                    matches += 1
            sim = matches / align_len if align_len > 0 else 0.0
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix, names


def condense_matrix(dist_matrix: np.ndarray) -> List[float]:
    n = dist_matrix.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(float(dist_matrix[i, j]))
    return condensed


def is_alignment_input(sequences: List[str]) -> bool:
    if not sequences:
        return False
    first_len = len(sequences[0])
    if first_len < 2:
        return False
    all_same_len = all(len(s) == first_len for s in sequences)
    has_gaps = any("-" in s for s in sequences)
    return all_same_len and has_gaps


def pair_align_all(
    sequences: List[str],
    names: List[str],
    mode: str = "needle",
    checkpoint_dir: str | None = None,
    resume: bool = False,
    progress_callback=None,
) -> Tuple[List[Tuple[str, str]], SimilarityMatrix]:
    engine = AlignmentEngine(mode=mode)
    sim_mat = compute_similarity_matrix(
        sequences, names, mode, checkpoint_dir, resume, progress_callback
    )
    aligned_pairs = engine.align_multiple(sequences, names)
    return aligned_pairs, sim_mat
