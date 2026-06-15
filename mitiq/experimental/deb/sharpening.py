# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Sharpening post-processing for debiasing.

Sharpening is a nonlinear aggregation step applied on top of the debiased
variants. Instead of averaging the variant distributions, it performs a
plurality vote shot by shot: for each shot index the bitstring that occurs
most often across the variants is kept. This concentrates probability on the
dominant outcome and is effective when the answer is supported on a small
number of bitstrings, e.g. optimization or eigenvalue problems. See
:cite:`Maksymov_2023_arxiv`.
"""

from collections import Counter
from collections.abc import Sequence

from mitiq import MeasurementResult


def sharpen(results: Sequence[MeasurementResult]) -> MeasurementResult:
    """Combine per-variant measurement results by shot-wise plurality voting.

    The variant results are expected to be aligned shot by shot. When variants
    have different shot counts the result is truncated to the smallest count.
    For each shot the bitstring that occurs most often across the variants is
    kept. A shot with no unique winner (a tie for the most common bitstring) is
    discarded, so the sharpened result may contain fewer shots than the inputs.

    Args:
        results: One ``MeasurementResult`` per symmetrized variant.

    Returns:
        A single ``MeasurementResult`` holding the surviving sharpened shots.
    """
    if not results:
        raise ValueError("sharpen requires at least one MeasurementResult.")

    num_shots = min(result.shots for result in results)

    sharpened: list[list[int]] = []
    for shot in range(num_shots):
        votes: list[tuple[int, ...]] = [
            tuple(int(bit) for bit in result.result[shot])
            for result in results
        ]
        ranking = Counter(votes).most_common(2)
        tie = len(ranking) > 1 and ranking[0][1] == ranking[1][1]
        if not tie:
            sharpened.append(list(ranking[0][0]))

    return MeasurementResult(sharpened)
