# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qiskit's circuit representation.
"""

import re
from typing import Any

import cirq
import numpy as np
import qiskit
from cirq.contrib.qasm_import import circuit_from_qasm
from cirq.contrib.qasm_import.exception import QasmException
from qiskit import qasm2
from qiskit.transpiler import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import SetLayout

from mitiq.interface.mitiq_qiskit.transpiler import (
    ApplyMitiqLayout,
    ClearLayout,
)
from mitiq.utils import _simplify_circuit_exponents

QASMType = str


def _remove_qasm_barriers(qasm: QASMType) -> QASMType:
    """Returns a copy of the input QASM with all barriers removed.

    Args:
        qasm: QASM to remove barriers from.

    Note:
        According to the OpenQASM 2.X language specification
        (https://arxiv.org/pdf/1707.03429v2.pdf), "Statements are separated by
        semicolons. Whitespace is ignored. The language is case sensitive.
        Comments begin with a pair of forward slashes and end with a new line."
    """
    quoted_re = r"(?:\"[^\"]*?\")"
    statement_re = r"((?:[^;{}\"]*?" + quoted_re + r"?)*[;{}])?"
    comment_re = r"(\n?//[^\n]*(?:\n|$))?"
    statements_comments = re.findall(statement_re + comment_re, qasm)
    lines = []
    for statement, comment in statements_comments:
        if re.match(r"^\s*barrier(?:(?:\s+)|(?:;))", statement) is None:
            lines.append(statement + comment)
    return "".join(lines)


def _map_bit_index(
    bit_index: int, new_register_sizes: list[int]
) -> tuple[int, int]:
    """Returns the register index and (qu)bit index in this register for the
    mapped bit_index.

    Args:
        bit_index: Index of (qu)bit in the original register.
        new_register_sizes: List of sizes of the new registers.

    Example:
        bit_index = 3, new_register_sizes = [2, 3]
        returns (1, 0), meaning the mapped (qu)bit is in the 1st new register
        and has index 0 in this register.

    Note:
        The bit_index is assumed to come from a circuit with 1 or n registers
        where n is the maximum bit_index.
    """
    max_indices_in_registers = np.cumsum(new_register_sizes) - 1

    # Could be faster via bisection.
    register_index = None
    for i in range(len(max_indices_in_registers)):
        if bit_index <= max_indices_in_registers[i]:
            register_index = i
            break
    assert register_index is not None

    if register_index == 0:
        return register_index, bit_index

    return (
        register_index,
        bit_index - max_indices_in_registers[register_index - 1] - 1,
    )


def _add_identity_to_idle(
    circuit: qiskit.QuantumCircuit,
) -> set[qiskit.circuit.Qubit]:
    """Adds identities to idle qubits in the circuit and returns the altered
    indices. Used to preserve idle qubits and indices in conversion.

    Args:
        circuit: Qiskit circuit to have identities added to idle qubits

    Returns:
        An unordered set of the indices that were altered

    Note: An idle qubit is a qubit without any gates (including Qiskit
        barriers) acting on it.
    """
    all_qubits = set(circuit.qubits)
    used_qubits = set()
    idle_qubits = set()
    # Get used qubits
    for op in circuit.data:
        qubits = op.qubits
        used_qubits.update(set(qubits))
    idle_qubits = all_qubits - used_qubits
    # Modify input circuit applying I to idle qubits
    for q in idle_qubits:
        circuit.id(q)

    return idle_qubits


def _remove_identity_from_idle(
    circuit: qiskit.QuantumCircuit,
    idle_qubits: set[qiskit.circuit.Qubit],
) -> None:
    """Removes identities from the circuit corresponding to the input
    idle qubits.
    Used in conjunction with _add_identity_to_idle to preserve idle qubits in
    conversion.

    Args:
        circuit: Qiskit circuit to have identities removed
        idle_indices: Set of altered idle qubits.
    """
    to_delete_indices: list[int] = []
    for index, op in enumerate(circuit._data):
        gate, qubits = op.operation, op.qubits
        if gate.name == "id" and set(qubits).intersection(idle_qubits):
            to_delete_indices.append(index)
    # Traverse data from list end to preserve index
    for index in to_delete_indices[::-1]:
        del circuit._data[index]


def _measurement_order(
    circuit: qiskit.QuantumCircuit,
) -> list[tuple[Any, ...]]:
    """Returns the left-to-right measurement order in the circuit.

    The "measurement order" is a list of tuples (qubit, bit) involved in
    measurements ordered as they appear going left-to-right through the circuit
    (i.e., iterating through circuit.data). The purpose of this is to be able
    to do

    >>> for (qubit, bit) in _measurement_order(circuit):
    >>>     other_circuit.measure(qubit, bit)

    which ensures ``other_circuit`` has the same measurement order as
    ``circuit``, assuming ``other_circuit`` has the same register(s) as
    ``circuit``.

    Args:
        circuit: Qiskit circuit to get the measurement order of.
    """
    order = []
    for op in circuit.data:
        gate, qubits, cbits = op.operation, op.qubits, op.clbits
        if isinstance(gate, qiskit.circuit.Measure):
            if len(qubits) != 1 or len(cbits) != 1:
                raise ValueError(
                    f"Only measurements with one qubit and one bit are "
                    f"supported, but this measurement has {len(qubits)} "
                    f"qubit(s) and {len(cbits)} bit(s). If you think this "
                    f"should be supported and is a bug, please open an issue "
                    f"at https://github.com/unitaryfoundation/mitiq."
                )
            order.append((*qubits, *cbits))
    return order


def _transform_registers(
    circuit: qiskit.QuantumCircuit,
    new_qregs: list[qiskit.QuantumRegister] | None = None,
) -> qiskit.QuantumCircuit:
    """Transforms the registers in the circuit to the new registers.

    Args:
        circuit: Qiskit circuit.
        new_qregs: The new quantum registers for the circuit.

    Raises:
        ValueError:
            * If the number of qubits in the new quantum registers is
            greater than the number of qubits in the circuit.
    """
    if new_qregs is None:
        return circuit

    qreg_sizes = [qreg.size for qreg in new_qregs]
    nqubits_in_circuit = circuit.num_qubits

    if len(qreg_sizes) and sum(qreg_sizes) < nqubits_in_circuit:
        raise ValueError(
            f"The circuit has {nqubits_in_circuit} qubit(s), but the provided "
            f"quantum registers have {sum(qreg_sizes)} qubit(s)."
        )

    circuit_layout = Layout.from_qubit_list(circuit.qubits)
    pass_manager = PassManager(
        [SetLayout(circuit_layout), ApplyMitiqLayout(new_qregs), ClearLayout()]
    )
    return pass_manager.run(circuit)


def to_qasm(circuit: cirq.Circuit) -> QASMType:
    """Returns a QASM string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a QASM string.

    Returns:
        QASMType: QASM string equivalent to the input Mitiq circuit.
    """
    # Simplify exponents of gates. For example, H**-1 is simplified to H.
    _simplify_circuit_exponents(circuit)
    return circuit.to_qasm()


def to_qiskit(circuit: cirq.Circuit) -> qiskit.QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit. Note
    that the output circuit registers may not match the input circuit
    registers.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    return qiskit.QuantumCircuit.from_qasm_str(to_qasm(circuit))


def from_qiskit(circuit: qiskit.QuantumCircuit) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input Qiskit circuit.

    Args:
        circuit: Qiskit circuit to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Qiskit circuit.
    """
    try:
        mitiq_circuit = from_qasm(qasm2.dumps(circuit))

    except QasmException:
        # Try to decompose circuit before running
        # This is necessary for converting qiskit circuits with
        # custom packaged gates, e.g., QFT gates
        BASIS_GATES = [
            "sx",
            "sxdg",
            "rx",
            "ry",
            "rz",
            "id",
            "u1",
            "u2",
            "u3",
            "r",
            "x",
            "y",
            "z",
            "h",
            "s",
            "t",
            "cx",
            "cy",
            "cz",
            "ch",
            "swap",
            "cswap",
            "ccx",
            "sdg",
            "tdg",
        ]
        circuit = qiskit.transpile(circuit, basis_gates=BASIS_GATES)
        mitiq_circuit = from_qasm(qasm2.dumps(circuit))
    return mitiq_circuit


def from_qasm(qasm: QASMType) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input QASM string.

    Args:
        qasm: QASM string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input QASM string.
    """
    qasm = _remove_qasm_barriers(qasm)
    return circuit_from_qasm(qasm)
