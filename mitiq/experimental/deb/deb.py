# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level debiasing functions.

Debiasing (also called symmetrization) mitigates qubit-dependent errors by
executing several variants of a circuit, each one relabeled onto a randomly
permuted set of qubits, and combining their results. Each variant's
permutation is undone on the measured bitstrings before they are combined.
The combined result is obtained either by averaging the variant distributions
or by a shot-wise plurality vote (sharpening). See :cite:`Maksymov_2023_arxiv`.
"""

from collections.abc import Callable, Sequence

import cirq
import numpy as np

from mitiq import MeasurementResult
from mitiq.experimental.deb.sharpening import sharpen
from mitiq.experimental.deb.symmetrization import _construct_variants

Executor = Callable[[cirq.Circuit], MeasurementResult]


def _unscramble(
    result: MeasurementResult, permutation: Sequence[int]
) -> MeasurementResult:
    """Undo a variant's qubit permutation on its measured bitstrings.

    Logical qubit ``i`` is measured on the qubit at index ``permutation[i]``,
    so the unscrambled bit for logical qubit ``i`` is ``shot[permutation[i]]``.
    """
    unscrambled = [
        [int(shot[permutation[i]]) for i in range(len(permutation))]
        for shot in result.result
    ]
    return MeasurementResult(unscrambled)


def _average(distributions: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average a list of probability distributions over the same basis."""
    total: dict[str, float] = {}
    for distribution in distributions:
        for bitstring, probability in distribution.items():
            total[bitstring] = total.get(bitstring, 0.0) + probability
    return {
        bitstring: probability / len(distributions)
        for bitstring, probability in total.items()
    }


def combine_results(
    results: Sequence[MeasurementResult],
    permutations: Sequence[Sequence[int]],
    method: str = "averaging",
) -> dict[str, float]:
    """Combine the measurement results of debiasing variants.

    Each result is first unscrambled with its variant's permutation. The
    unscrambled results are then combined either by averaging their probability
    distributions (``"averaging"``) or by a shot-wise plurality vote
    (``"sharpening"``, see :func:`.sharpen`).

    Args:
        results: One ``MeasurementResult`` per variant, in the same order as
            the variants returned by :func:`.construct_circuits`.
        permutations: The permutation applied to each variant, in the same
            order as ``results``.
        method: Either ``"averaging"`` or ``"sharpening"``.

    Returns:
        The combined probability distribution over the computational basis.
    """
    unscrambled = [
        _unscramble(result, permutation)
        for result, permutation in zip(results, permutations)
    ]

    if method == "averaging":
        return _average([result.prob_distribution() for result in unscrambled])
    if method == "sharpening":
        sharpened = sharpen(unscrambled)
        if sharpened.shots == 0:
            return {}
        return sharpened.prob_distribution()

    raise ValueError(
        f"Unknown method {method!r}. Use 'averaging' or 'sharpening'."
    )


def execute_with_debiasing(
    circuit: cirq.Circuit,
    executor: Executor,
    num_variants: int = 10,
    *,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, float]:
    """Return the debiased probability distribution of ``circuit``.

    The circuit is relabeled onto ``num_variants`` randomly permuted sets of
    qubits, each variant is executed, its permutation is undone on the measured
    bitstrings, and the resulting probability distributions are averaged.

    Args:
        circuit: The circuit to execute. It should not contain terminal
            measurements; the executor is responsible for measurement.
        executor: A function mapping a circuit to a ``MeasurementResult``.
        num_variants: Number of permuted variants to average over.
        random_state: Seed or ``numpy.random.Generator`` for reproducibility.

    Returns:
        The averaged probability distribution over the computational basis.
    """
    variants = _construct_variants(circuit, num_variants, random_state)
    results = [executor(variant) for variant, _ in variants]
    permutations = [permutation for _, permutation in variants]
    return combine_results(results, permutations, method="averaging")


def execute_with_debiasing_and_sharpening(
    circuit: cirq.Circuit,
    executor: Executor,
    num_variants: int = 10,
    *,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, float]:
    """Return the debiased and sharpened probability distribution.

    Like :func:`execute_with_debiasing`, but the unscrambled variant results
    are combined with a shot-wise plurality vote (see :func:`.sharpen`) instead
    of being averaged. This is useful when the target answer is concentrated on
    a few bitstrings.

    Args:
        circuit: The circuit to execute. It should not contain terminal
            measurements; the executor is responsible for measurement.
        executor: A function mapping a circuit to a ``MeasurementResult``.
        num_variants: Number of permuted variants to combine.
        random_state: Seed or ``numpy.random.Generator`` for reproducibility.

    Returns:
        The sharpened probability distribution over the computational basis.
    """
    variants = _construct_variants(circuit, num_variants, random_state)
    results = [executor(variant) for variant, _ in variants]
    permutations = [permutation for _, permutation in variants]
    return combine_results(results, permutations, method="sharpening")
