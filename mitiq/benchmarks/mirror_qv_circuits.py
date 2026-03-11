# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to create a Mirror Quantum Volume Benchmarking circuit
as defined in https://arxiv.org/abs/2303.02108."""

from typing import cast

import cirq

from mitiq import QPROGRAM
from mitiq.benchmarks.quantum_volume_circuits import (
    generate_quantum_volume_circuit,
)
from mitiq.interface.conversions import convert_from_mitiq


def generate_mirror_qv_circuit(
    num_qubits: int,
    depth: int,
    decompose: bool = False,
    seed: int | None = None,
    return_type: str | None = None,
) -> QPROGRAM:
    """Generate a mirror quantum volume circuit with the given number of qubits
    and depth as defined in :cite:`Amico_2023_arxiv`.

    The generated circuit consists of a quantum volume circuit up to `depth/2`
    layers followed by an inverse of the quantum volume portion up to `depth/2`
    when `depth` is an even number.

    When `depth` is odd, the layers will be changed to `depth+1`.


    The ideal output bit-string is a string of zeroes.

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of layers in the generated circuit.
        decompose: Recursively decomposes the randomly sampled (numerical)
            unitary matrix gates into simpler gates.
        seed: Seed for generating random circuit.
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A quantum volume circuit acting on ``num_qubits`` qubits.
    """
    if depth <= 0:
        raise ValueError(
            "{} is invalid for the generated circuit depth.", depth
        )

    if depth % 2 == 0:
        first_half_depth = int(depth / 2)
    else:
        first_half_depth = int((depth + 1) / 2)

    qv_circuit, _ = generate_quantum_volume_circuit(
        num_qubits, first_half_depth, seed=seed, decompose=decompose
    )
    qv_circuit = cast(cirq.Circuit, qv_circuit)

    mirror_qv_circuit = qv_circuit + cirq.inverse(qv_circuit)

    if decompose:
        # Decompose random unitary gates into simpler gates.
        mirror_qv_circuit = cirq.Circuit(cirq.decompose(mirror_qv_circuit))

    return_type = "cirq" if not return_type else return_type
    return convert_from_mitiq(mirror_qv_circuit, return_type)
