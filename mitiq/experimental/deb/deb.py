# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level debiasing functions.

Debiasing (also called symmetrization) mitigates qubit-dependent errors by
executing several variants of a circuit, each one relabeled onto a randomly
permuted set of qubits, and combining their results. Each variant's
permutation is undone on the measured bitstrings before they are combined.
Averaging the unscrambled distributions reduces qubit-dependent bias; an
optional sharpening step instead keeps the shot-wise majority outcome.
See :cite:`Maksymov_2023_arxiv`.
"""

import random
from collections.abc import Callable

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import MeasurementResult
from mitiq.experimental.deb.sharpening import sharpen
from mitiq.experimental.deb.symmetrization import _construct_variants

Executor = Callable[[cirq.Circuit], MeasurementResult]


def _unscramble(
    result: MeasurementResult, permutation: list[int]
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


def _probability_vector(
    result: MeasurementResult, num_qubits: int
) -> npt.NDArray[np.float64]:
    """Return the probability vector (length ``2 ** num_qubits``) of a
    measurement result."""
    vector = np.zeros(2**num_qubits)
    for bitstring, probability in result.prob_distribution().items():
        vector[int(bitstring, 2)] += probability
    return vector


def execute_with_debiasing(
    circuit: cirq.Circuit,
    executor: Executor,
    num_variants: int = 10,
    *,
    random_state: int | random.Random | None = None,
) -> npt.NDArray[np.float64]:
    """Return the debiased probability distribution of ``circuit``.

    The circuit is relabeled onto ``num_variants`` randomly permuted sets of
    qubits, each variant is executed, its permutation is undone on the measured
    bitstrings, and the resulting probability distributions are averaged.

    Args:
        circuit: The circuit to execute. It should not contain terminal
            measurements; the executor is responsible for measurement.
        executor: A function mapping a circuit to a ``MeasurementResult``.
        num_variants: Number of permuted variants to average over.
        random_state: Seed or ``random.Random`` for reproducible sampling.

    Returns:
        The averaged probability vector over the computational basis.
    """
    num_qubits = len(circuit.all_qubits())
    variants = _construct_variants(circuit, num_variants, random_state)

    distributions = [
        _probability_vector(
            _unscramble(executor(variant), permutation), num_qubits
        )
        for variant, permutation in variants
    ]
    return np.mean(distributions, axis=0)


def execute_with_debiasing_and_sharpening(
    circuit: cirq.Circuit,
    executor: Executor,
    num_variants: int = 10,
    *,
    random_state: int | random.Random | None = None,
) -> npt.NDArray[np.float64]:
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
        random_state: Seed or ``random.Random`` for reproducible sampling.

    Returns:
        The sharpened probability vector over the computational basis.
    """
    num_qubits = len(circuit.all_qubits())
    variants = _construct_variants(circuit, num_variants, random_state)

    results = [
        _unscramble(executor(variant), permutation)
        for variant, permutation in variants
    ]
    return _probability_vector(sharpen(results), num_qubits)
