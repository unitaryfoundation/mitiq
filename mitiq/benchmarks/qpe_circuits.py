# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to create a QPE circuit."""

import cirq

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_qpe_circuit(
    evalue_reg: int,
    input_gate: cirq.Gate = cirq.T,
    return_type: str | None = None,
) -> QPROGRAM:
    """Returns a circuit to create a quantum phase estimation (QPE) circuit as
    defined in https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm

    The unitary to estimate the phase of corresponds to a
    single-qubit gate (``input_gate``).

    The IQFT circuit defined in this method is taken from taken from Sec 7.7.4
    of :cite:`Wong_2022`. The notation for eigenvalue register and eigenstate
    register used to define this function also follows from :cite:`Wong_2022`.

    Args:
        evalue_reg : Number of qubits in the eigenvalue register. The qubits
            in this variable are used to estimate the phase.
        input_gate : The unitary to estimate the phase of as a single-qubit
            Cirq gate. Default gate used here is `cirq.T`.
        return_type: Return type of the output circuit.
    Returns:
        A Quantum Phase Estimation circuit.
    """
    if evalue_reg <= 0:
        raise ValueError(
            "{} is invalid for the number of eigenvalue reg qubits. ",
            evalue_reg,
        )
    num_qubits_for_gate = input_gate.num_qubits()
    if num_qubits_for_gate > 1:
        raise ValueError("This QPE method only works for 1-qubit gates.")

    if evalue_reg == num_qubits_for_gate:
        raise ValueError(
            "The eigenvalue reg must be larger than the eigenstate reg."
        )

    total_num_qubits = evalue_reg + num_qubits_for_gate
    qreg = cirq.LineQubit.range(total_num_qubits)
    circuit = cirq.Circuit()

    # QFT circuit
    # apply hadamard and controlled unitary to the qubits in the eigenvalue reg
    hadamard_circuit = cirq.Circuit()
    for i in range(evalue_reg):
        hadamard_circuit.append(cirq.H(qreg[i]))
    circuit = circuit + hadamard_circuit

    for i in range(total_num_qubits - 1)[::-1]:
        circuit.append(
            [input_gate(qreg[-1]).controlled_by(qreg[i])]
            * (2 ** (evalue_reg - 1 - i))
        )

    # IQFT of the eigenvalue register
    # swap the qubits in the eigenvalue register
    for i in range(int(evalue_reg / 2)):
        circuit.append(
            cirq.SWAP(qreg[i], qreg[evalue_reg - 1 - i]),
            strategy=cirq.InsertStrategy.NEW,
        )
    # apply inverse of hadamard followed by controlled unitary
    circuit.append(cirq.H(qreg[0]), strategy=cirq.InsertStrategy.NEW)
    for i in range(1, evalue_reg):
        for j in range(evalue_reg):
            if j < i:
                circuit.append(
                    cirq.inverse(input_gate(qreg[i]).controlled_by(qreg[j])),
                    strategy=cirq.InsertStrategy.NEW,
                )
        circuit.append(
            cirq.H(qreg[i]),
            strategy=cirq.InsertStrategy.NEW,
        )
    return_type = "cirq" if not return_type else return_type

    return convert_from_mitiq(circuit, return_type)
