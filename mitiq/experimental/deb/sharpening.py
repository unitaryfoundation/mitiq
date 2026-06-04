# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Sharpening utilities for debiased bitstring distributions."""

from collections import Counter
from collections.abc import Sequence

from mitiq import MeasurementResult

BitstringDistribution = dict[str, int | float]


def sharpen(
    results: Sequence[BitstringDistribution | MeasurementResult],
) -> dict[str, float]:
    """Applies plurality-vote sharpening to variant bitstring results.

    For each symmetrized variant, the most likely bitstring is selected. The
    returned distribution is the empirical distribution of those per-variant
    winners. Ties are resolved lexicographically so the output is
    deterministic.

    Args:
        results: Count dictionaries, probability dictionaries, or Mitiq
            ``MeasurementResult`` objects, one per symmetrized variant.

    Raises:
        ValueError: If no results are provided or a result is empty.

    Returns:
        A normalized bitstring probability distribution after sharpening.
    """
    if not results:
        raise ValueError("At least one result is required for sharpening.")

    winners = Counter[str]()
    for result in results:
        distribution = _as_distribution(result)
        if not distribution:
            raise ValueError("Cannot sharpen an empty result.")
        winner = max(
            sorted(distribution),
            key=lambda bitstring: distribution[bitstring],
        )
        winners[winner] += 1

    total = float(sum(winners.values()))
    return {
        bitstring: count / total
        for bitstring, count in sorted(winners.items())
    }


def _as_distribution(
    result: BitstringDistribution | MeasurementResult,
) -> dict[str, float]:
    """Converts supported result containers to normalized distributions."""
    if isinstance(result, MeasurementResult):
        return result.prob_distribution()

    total = float(sum(result.values()))
    if total <= 0:
        raise ValueError("Result weights must sum to a positive value.")
    return {
        bitstring: float(weight) / total
        for bitstring, weight in result.items()
    }
