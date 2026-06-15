# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Construction of permuted (debiasing) circuit variants.

Debiasing reduces the effect of qubit-dependent errors by running several
variants of a circuit, each one with the qubits relabeled by a random
permutation. A variant maps the circuit ``C`` onto a permuted set of qubits,
``C -> pi C`` for a permutation ``pi``. On a noiseless device the permutation
is undone in post-processing, so every variant reproduces the ideal result.
On hardware each variant samples a different assignment of logical qubits to
physical qubits, so qubit-dependent biases are averaged out when the results
are combined. See :cite:`Maksymov_2023_arxiv`.
"""

import cirq
import numpy as np


def _random_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Return a ``numpy`` generator from a seed, an existing generator, or
    ``None``."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _permute(
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    permutation: list[int],
) -> cirq.Circuit:
    """Relabel ``circuit`` so the operation on ``qubits[i]`` acts on
    ``qubits[permutation[i]]``."""
    qubit_map = {qubits[i]: qubits[permutation[i]] for i in range(len(qubits))}
    return circuit.transform_qubits(qubit_map)


def _construct_variants(
    circuit: cirq.Circuit,
    num_variants: int,
    random_state: int | np.random.Generator | None,
) -> list[tuple[cirq.Circuit, list[int]]]:
    """Return ``(variant, permutation)`` pairs for debiasing.

    ``permutation[i]`` is the index of the qubit that logical qubit ``i`` is
    mapped onto, which is also what post-processing uses to unscramble the
    measured bitstrings.
    """
    if num_variants < 1:
        raise ValueError("num_variants must be a positive integer.")

    qubits = sorted(circuit.all_qubits())
    rng = _random_generator(random_state)

    variants = []
    for _ in range(num_variants):
        permutation = [int(i) for i in rng.permutation(len(qubits))]
        variants.append((_permute(circuit, qubits, permutation), permutation))

    return variants


def construct_circuits(
    circuit: cirq.Circuit,
    num_variants: int = 10,
    *,
    random_state: int | np.random.Generator | None = None,
) -> list[cirq.Circuit]:
    r"""Return permuted variants of ``circuit`` for debiasing.

    Each variant relabels the circuit onto a randomly permuted set of qubits,
    ``C -> pi C``. The input circuit should not contain terminal measurements;
    measurement is expected to be handled by the executor. To recover the ideal
    result, the permutation applied to each variant must be undone on the
    measured bitstrings (see :func:`.execute_with_debiasing`).

    Args:
        circuit: The circuit to permute.
        num_variants: Number of permuted variants to generate.
        random_state: Seed or ``numpy.random.Generator`` for reproducible
            sampling of the permutations.

    Returns:
        A list of ``num_variants`` permuted copies of ``circuit``.
    """
    return [
        variant
        for variant, _ in _construct_variants(
            circuit, num_variants, random_state
        )
    ]
