from __future__ import annotations

from typing import List, Tuple

from Bio.Align import PairwiseAligner, substitution_matrices


class AlignmentEngine:
    def __init__(
        self,
        mode: str = "global",
        match: int = 2,
        mismatch: int = -1,
        gap_open: int = -5,
        gap_extend: int = -0.5,
        matrix_name: str | None = None,
    ):
        self.aligner = PairwiseAligner()
        if mode == "needle":
            self.aligner.mode = "global"
        elif mode == "water":
            self.aligner.mode = "local"
        else:
            self.aligner.mode = mode
        if matrix_name:
            try:
                sub_mat = substitution_matrices.load(matrix_name)
                self.aligner.substitution_matrix = sub_mat
            except Exception:
                self.aligner.match_score = match
                self.aligner.mismatch_score = mismatch
        else:
            self.aligner.match_score = match
            self.aligner.mismatch_score = mismatch
        self.aligner.open_gap_score = gap_open
        self.aligner.extend_gap_score = gap_extend

    def align_pair(
        self, seq1: str, seq2: str
    ) -> Tuple[str, str, float]:
        alignments = self.aligner.align(seq1, seq2)
        if not alignments:
            return seq1, seq2, 0.0
        best = alignments[0]
        score = best.score
        aligned_str = str(best).split("\n")
        if len(aligned_str) >= 3:
            aligned_seq1 = aligned_str[0]
            aligned_seq2 = aligned_str[2]
        else:
            aligned_seq1 = seq1
            aligned_seq2 = seq2
        max_possible = max(len(seq1), len(seq2))
        if self.aligner.mode == "global":
            max_possible = max(len(seq1), len(seq2)) * max(
                self.aligner.match_score, 2
            )
            if max_possible > 0:
                similarity = max(0.0, min(1.0, score / max_possible))
            else:
                similarity = 0.0
        else:
            similarity = self._compute_similarity(aligned_seq1, aligned_seq2)
        return aligned_seq1, aligned_seq2, similarity

    def align_pair_simple(
        self, seq1: str, seq2: str
    ) -> float:
        alignments = self.aligner.align(seq1, seq2)
        if not alignments:
            return 0.0
        best = alignments[0]
        aligned_str = str(best).split("\n")
        if len(aligned_str) >= 3:
            aligned_seq1 = aligned_str[0]
            aligned_seq2 = aligned_str[2]
            return self._compute_similarity(aligned_seq1, aligned_seq2)
        return 0.0

    @staticmethod
    def _compute_similarity(aligned1: str, aligned2: str) -> float:
        matches = 0
        total = 0
        for c1, c2 in zip(aligned1, aligned2):
            if c1 == "-" or c2 == "-":
                total += 1
            else:
                total += 1
                if c1 == c2:
                    matches += 1
        if total == 0:
            return 0.0
        return matches / total

    def align_multiple(
        self, sequences: List[str], names: List[str]
    ) -> List[Tuple[str, str]]:
        if len(sequences) <= 1:
            return list(zip(names, sequences))
        ref_idx = 0
        max_len = len(sequences[0])
        for i, s in enumerate(sequences):
            if len(s) > max_len:
                max_len = len(s)
                ref_idx = i
        ref_seq = sequences[ref_idx]
        aligned_pairs = []
        ref_aligned, _, _ = self.align_pair(ref_seq, ref_seq)
        aligned_pairs.append((names[ref_idx], ref_seq))
        for i, seq in enumerate(sequences):
            if i == ref_idx:
                continue
            _, aligned_seq, _ = self.align_pair(ref_seq, seq)
            aligned_pairs.append((names[i], aligned_seq))
        return aligned_pairs
