# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qrisp's circuit representation.
"""

from cirq import Circuit
from qrisp import QuantumCircuit


def from_qrisp(circuit: QuantumCircuit) -> Circuit:
    """Convert a Qrisp QuantumCircuit to a cirq Circuit.
    Args:
        circuit (QuantumCircuit): The Qrisp QuantumCircuit to convert.
    Returns:
        Circuit: The converted cirq Circuit.
    """
    from qiskit import QuantumCircuit as QiskitCircuit

    from mitiq.interface.mitiq_qiskit import from_qiskit

    # To avoid qubit naming inconsistencies, we convert qrisp circuit
    # to qiskit circuit first, then compose that to an empty
    # qiskit circuit with default qubit naming
    qrisp_to_qiskit = circuit.to_qiskit()
    circuit = QiskitCircuit(
        qrisp_to_qiskit.num_qubits, qrisp_to_qiskit.num_clbits
    )

    circuit.compose(
        qrisp_to_qiskit,
        range(qrisp_to_qiskit.num_qubits),
        range(qrisp_to_qiskit.num_clbits),
        inplace=True,
    )

    return from_qiskit(circuit)


def to_qrisp(circuit: Circuit) -> QuantumCircuit:
    """Convert a cirq Circuit to a Qrisp QuantumCircuit.
    Args:
        circuit (Circuit): The cirq Circuit to convert.
    Returns:
        QuantumCircuit: The converted Qrisp QuantumCircuit.
    """
    from mitiq.interface.mitiq_qiskit import to_qiskit

    qiskit_circuit = to_qiskit(circuit)
    return QuantumCircuit.from_qiskit(qiskit_circuit)
