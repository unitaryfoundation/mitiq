# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for creating a linear complexity W-state benchmarking circuit
as defined in :cite:`Cruz_2019_Efficient`."""

import cirq
import numpy as np

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_w_circuit(
    n_qubits: int,
    return_type: str | None = None,
) -> QPROGRAM:
    """Returns a circuit to create a ``n_qubits`` qubit Werner-state with
    linear complexity as defined in :cite:`Cruz_2019_Efficient`.


    Args:
        n_qubits : The number of qubits in the circuit.
        return_type : Return type of the output circuit.
    Returns:
        A W-state circuit of linear complexity acting on ``n_qubits`` qubits.
    """
    if n_qubits <= 0:
        raise ValueError("{} is invalid for the number of qubits. ", n_qubits)

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    for i, j in zip(range(0, n_qubits), range(1, n_qubits)):
        N = n_qubits - i
        angle = 2 * np.arccos(np.sqrt(1 / N))
        circuit.append(
            cirq.Ry(rads=angle).controlled().on(qubits[i], qubits[j])
        )
        circuit.append(cirq.CNOT(qubits[j], qubits[i]))

    return_type = "cirq" if not return_type else return_type

    return convert_from_mitiq(circuit, return_type)
