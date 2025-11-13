# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Pennylane's circuit representation.
"""

import pennylane as qml
from cirq import Circuit
from pennylane import from_qasm as pennylane_from_qasm
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP

from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm
from mitiq.interface.mitiq_qiskit import to_qasm

SUPPORTED_PL = set(QISKIT_OPERATION_MAP.keys())
UNSUPPORTED = {"CRX", "CRY", "CRZ", "S", "T"}
SUPPORTED = SUPPORTED_PL - UNSUPPORTED


class UnsupportedQuantumTapeError(Exception):
    pass


def from_pennylane(tape: QuantumTape) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input QuantumTape.

    Args:
        tape: Pennylane QuantumTape to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input QuantumTape.
    """
    try:
        wires = sorted(tape.wires)
    except TypeError:
        raise UnsupportedQuantumTapeError(
            f"The wires of the tape must be sortable, but could not sort "
            f"{tape.wires}."
        )

    for i in range(len(wires)):
        if wires[i] != i:
            raise UnsupportedQuantumTapeError(
                "The wire labels of the tape must contiguously pack 0 "
                "to n-1, for n wires."
            )

    if len(tape.measurements) > 0:
        raise UnsupportedQuantumTapeError(
            "Measurements are not supported on the input tape. "
            "They should be subsequently added by the executor."
        )

    tape = tape.expand(stop_at=lambda obj: obj.name in SUPPORTED)
    qasm = qml.to_openqasm(
        tape,
        rotations=False,
        wires=wires,
        measure_all=False,
    )

    return cirq_from_qasm(qasm)


def to_pennylane(circuit: Circuit) -> QuantumTape:
    """Returns a QuantumTape equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Pennylane QuantumTape.

    Returns:
        QuantumTape object equivalent to the input Mitiq circuit.
    """
    qfunc = pennylane_from_qasm(to_qasm(circuit))

    with QuantumTape() as tape:
        qfunc(wires=Wires(range(len(circuit.all_qubits()))))

    return tape
