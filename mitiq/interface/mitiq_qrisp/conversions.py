# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qrisp's circuit representation.
"""

from cirq import Circuit
from qrisp import QuantumCircuit as QrispCircuit
from qrisp.interface import convert_from_cirq, convert_to_cirq


def from_qrisp(qrisp_circuit: QrispCircuit) -> Circuit:
    """Returns a Cirq circuit equivalent to the input Qrisp circuit.

    Args:
        qrisp_circuit: Qrisp circuit to convert to a Cirq circuit.

    Returns:
        Cirq circuit representation equivalent to the input Qrisp circuit.
    """
    return convert_to_cirq(qrisp_circuit)


def to_qrisp(circuit: Circuit) -> QrispCircuit:
    """Returns a Qrisp circuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a Qrisp circuit.

    Returns:
        QrispCircuit object equivalent to the input Mitiq circuit.
    """
    return convert_from_cirq(circuit)
