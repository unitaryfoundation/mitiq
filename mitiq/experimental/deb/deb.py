# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level debiasing functions.

Debiasing (also called symmetrization) mitigates coherent errors by executing
several Pauli-symmetrized variants of a circuit and combining their results.
Averaging the variant distributions turns coherent errors into incoherent
ones; an optional sharpening step instead keeps the shot-wise majority outcome.
See :cite:`Maksymov_2023_arxiv`.
"""

import random
from collections.abc import Callable

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import MeasurementResult
from mitiq.experimental.deb.sharpening import sharpen
from mitiq.experimental.deb.symmetrization import construct_circuits

Executor = Callable[[cirq.Circuit], MeasurementResult]


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

    The circuit is symmetrized into ``num_variants`` Pauli-conjugated variants,
    each is executed, and the resulting probability distributions are averaged.

    Args:
        circuit: The circuit to execute. It should not contain terminal
            measurements; the executor is responsible for measurement.
        executor: A function mapping a circuit to a ``MeasurementResult``.
        num_variants: Number of symmetrized variants to average over.
        random_state: Seed or ``random.Random`` for reproducible sampling.

    Returns:
        The averaged probability vector over the computational basis.
    """
    qubits = sorted(circuit.all_qubits())
    variants = construct_circuits(
        circuit, num_variants, random_state=random_state
    )

    distributions = [
        _probability_vector(executor(variant), len(qubits))
        for variant in variants
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

    Like :func:`execute_with_debiasing`, but the variant results are combined
    with a shot-wise plurality vote (see :func:`.sharpen`) instead of being
    averaged. This is useful when the target answer is concentrated on a few
    bitstrings.

    Args:
        circuit: The circuit to execute. It should not contain terminal
            measurements; the executor is responsible for measurement.
        executor: A function mapping a circuit to a ``MeasurementResult``.
        num_variants: Number of symmetrized variants to combine.
        random_state: Seed or ``random.Random`` for reproducible sampling.

    Returns:
        The sharpened probability vector over the computational basis.
    """
    qubits = sorted(circuit.all_qubits())
    variants = construct_circuits(
        circuit, num_variants, random_state=random_state
    )

    results = [executor(variant) for variant in variants]
    return _probability_vector(sharpen(results), len(qubits))
