# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Construction of symmetrized (debiasing) circuit variants.

Debiasing reduces the effect of coherent errors by running many variants of
a circuit that are equivalent in the noiseless case but accumulate coherent
errors differently. Each variant is built by conjugating the circuit with a
layer of random single-qubit Pauli gates: the same Pauli is applied to a
qubit before and after the circuit. Since Pauli gates are self-inverse the
layer cancels for an ideal execution, so no extra qubits or gates are needed
beyond the pre/post Pauli layer. See :cite:`Maksymov_2023_arxiv`.
"""

import random

import cirq

# Single-qubit Paulis sampled (uniformly) to build each symmetrization layer.
_PAULIS = (cirq.I, cirq.X, cirq.Y, cirq.Z)


def _random_generator(
    random_state: int | random.Random | None,
) -> random.Random:
    """Return a ``random.Random`` from a seed, an existing generator, or
    ``None``."""
    if isinstance(random_state, random.Random):
        return random_state
    return random.Random(random_state)


def construct_circuits(
    circuit: cirq.Circuit,
    num_variants: int = 10,
    *,
    random_state: int | random.Random | None = None,
) -> list[cirq.Circuit]:
    r"""Return symmetrized variants of ``circuit`` for debiasing.

    Each variant prepends and appends the same layer of random single-qubit
    Pauli gates, sampled uniformly from :math:`\{I, X, Y, Z\}` per qubit. The
    input circuit should not contain terminal measurements; measurement is
    expected to be handled by the executor.

    Args:
        circuit: The circuit to symmetrize.
        num_variants: Number of Pauli-symmetrized variants to generate.
        random_state: Seed or ``random.Random`` instance for reproducible
            sampling of the Pauli layers.

    Returns:
        A list of ``num_variants`` symmetrized copies of ``circuit``.
    """
    if num_variants < 1:
        raise ValueError("num_variants must be a positive integer.")

    qubits = sorted(circuit.all_qubits())
    rng = _random_generator(random_state)

    variants = []
    for _ in range(num_variants):
        layer = [rng.choice(_PAULIS) for _ in qubits]
        pauli_moment = cirq.Circuit(
            gate.on(qubit)
            for gate, qubit in zip(layer, qubits)
            if gate is not cirq.I
        )
        variants.append(pauli_moment + circuit + pauli_moment)

    return variants
