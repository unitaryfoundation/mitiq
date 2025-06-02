# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_random_clifford_t_circuit(
    num_qubits: int,
    num_oneq_cliffords: int,
    num_twoq_cliffords: int,
    num_t_gates: int,
    return_type: str | None = None,
    seed: int | None = None,
) -> QPROGRAM:
    r"""Generate a random quantum circuit with the given number of qubits,
    number of one-qubit Cliffords, number of two-qubit Cliffords and number
    of T gates.

    Args:
        num_qubits: The number of qubits in the generated circuit.
        num_oneq_cliffords: Number of one-qubit Cliffords to be used.
        num_twoq_cliffords: Number of two-qubit Cliffords to be used.
        num_t_gates: Number of T gates to be used.
        seed: Seed for generating random circuit.
        return_type: String which specifies the type of the returned
            circuits. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuits have type
            ``cirq.Circuit``.

    Returns:
        A quantum circuit acting on ``num_qubits`` qubits.
    """

    if num_qubits <= 0:
        raise ValueError(
            "Cannot prepare a circuit with {} qubits.", num_qubits
        )
    elif num_qubits == 1 and num_twoq_cliffords > 0:
        raise ValueError(
            "Need more than 2 qubits for two-qubit Clifford gates."
        )

    rnd_state = np.random.RandomState(seed)

    oneq_cliffords = [cirq.S, cirq.H]
    twoq_cliffords = [cirq.CNOT, cirq.CZ]

    oneq_idx_list = rnd_state.choice(len(oneq_cliffords), num_oneq_cliffords)
    twoq_idx_list = rnd_state.choice(len(twoq_cliffords), num_twoq_cliffords)

    oneq_list = [oneq_cliffords[i] for i in oneq_idx_list]
    twoq_list = [twoq_cliffords[i] for i in twoq_idx_list]

    t_list = [cirq.T for _ in range(num_t_gates)]

    all_gates = oneq_list + twoq_list + t_list
    rnd_state.shuffle(all_gates)

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

    qubits_idx = list(range(num_qubits))

    for gate in all_gates:
        qubits_for_gate_idx = rnd_state.choice(
            qubits_idx, size=gate.num_qubits(), replace=False
        )
        qubits_for_gate = [qubits[i] for i in qubits_for_gate_idx]
        operation = gate.on(*qubits_for_gate)
        circuit.append(operation)

    return_type = "cirq" if not return_type else return_type
    return convert_from_mitiq(circuit, return_type)
