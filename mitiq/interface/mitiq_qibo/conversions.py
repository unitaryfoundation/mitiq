# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Qibo's circuit representation.
"""

from cirq import Circuit
from numpy import pi
from qibo import gates
from qibo.gates.abstract import Gate, ParametrizedGate
from qibo.models.circuit import Circuit as QiboCircuit

from mitiq.interface.mitiq_qiskit import from_qasm as cirq_from_qasm
from mitiq.interface.mitiq_qiskit import to_qasm as cirq_to_qasm


def crx_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CRX gate to Cirq known gates.

    Args:
        qibo_gate: CRX gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RZ(q1, pi / 2),
        gates.RY(q1, theta / 2),
        gates.CNOT(q0, q1),
        gates.RY(q1, -theta / 2),
        gates.CNOT(q0, q1),
        gates.RZ(q1, -pi / 2),
    ]
    return decomp_gate


def cry_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CRY gate to Cirq known gates.

    Args:
        gate: CRY gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RY(q1, theta / 2),
        gates.CNOT(q0, q1),
        gates.RY(q1, -theta / 2),
        gates.CNOT(q0, q1),
    ]
    return decomp_gate


def crz_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CRZ gate to Cirq known gates.

    Args:
        gate: CRZ gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RZ(q1, theta / 2),
        gates.CNOT(q0, q1),
        gates.RZ(q1, -theta / 2),
        gates.CNOT(q0, q1),
    ]
    return decomp_gate


def cu1_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CU1 gate to Cirq known gates.

    Args:
        gate: CU1 gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.U1(q0, theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, -theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, theta / 2),
    ]
    return decomp_gate


def cu3_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CU3 gate to Cirq known gates.

    Args:
        gate: CU3 gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    phi = gate.parameters[1]
    lam = gate.parameters[2]
    decomp_gate = [
        gates.U1(q0, lam / 2 + phi / 2),
        gates.U1(q1, lam / 2 - phi / 2),
        gates.CNOT(q0, q1),
        gates.U3(q1, -theta / 2, 0, -lam / 2 - phi / 2),
        gates.CNOT(q0, q1),
        gates.U3(q1, theta / 2, phi, 0),
    ]
    return decomp_gate


def csx_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CSX gate to Cirq known gates.

    Args:
        gate: CSX gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = pi / 2
    decomp_gate = [
        gates.H(q1),
        gates.U1(q0, theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, -theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, theta / 2),
        gates.H(q1),
    ]
    return decomp_gate


def csxdg_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes CSXDG gate to Cirq known gates.

    Args:
        gate: CSXDG gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = -pi / 2
    decomp_gate = [
        gates.H(q1),
        gates.U1(q0, theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, -theta / 2),
        gates.CNOT(q0, q1),
        gates.U1(q1, theta / 2),
        gates.H(q1),
    ]
    return decomp_gate


def iswap_decomp(gate: Gate) -> list[Gate]:
    """Decomposes ISWAP gate to Cirq known gates.

    Args:
        gate: ISWAP gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    decomp_gate = [
        gates.S(q1),
        gates.S(q0),
        gates.H(q0),
        gates.CNOT(q0, q1),
        gates.CNOT(q1, q0),
        gates.H(q1),
    ]
    return decomp_gate


def fswap_decomp(gate: Gate) -> list[Gate]:
    """Decomposes FSWAP gate to Cirq known gates.

    Args:
        gate: FSWAP gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    decomp_gate = [
        gates.H(q1),
        gates.H(q0),
        gates.CNOT(q1, q0),
        gates.RZ(q0, pi / 2),
        gates.CNOT(q1, q0),
        gates.H(q1),
        gates.H(q0),
        gates.RX(q0, pi / 2),
        gates.RX(q1, pi / 2),
        gates.CNOT(q1, q0),
        gates.RZ(q0, pi / 2),
        gates.CNOT(q1, q0),
        gates.RX(q0, -pi / 2),
        gates.RX(q1, -pi / 2),
        gates.RZ(q0, pi / 2),
        gates.RZ(q1, pi / 2),
    ]
    return decomp_gate


def rxx_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes RXX gate to Cirq known gates.

    Args:
        gate: RXX gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.H(q0),
        gates.H(q1),
        gates.CNOT(q0, q1),
        gates.RZ(q1, theta),
        gates.CNOT(q0, q1),
        gates.H(q0),
        gates.H(q1),
    ]
    return decomp_gate


def ryy_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes RYY gate to Cirq known gates.

    Args:
        gate: RYY gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.RX(q0, pi / 2),
        gates.RX(q1, pi / 2),
        gates.CNOT(q0, q1),
        gates.RZ(q1, theta),
        gates.CNOT(q0, q1),
        gates.RX(q0, -pi / 2),
        gates.RX(q1, -pi / 2),
    ]
    return decomp_gate


def rzz_decomp(gate: ParametrizedGate) -> list[Gate]:
    """Decomposes RZZ gate to Cirq known gates.

    Args:
        gate: RZZ gate to decompose.

    Returns:
        List with gates that has the same effect as applying the original gate.
    """
    q0, q1 = gate.init_args
    theta = gate.parameters[0]
    decomp_gate = [
        gates.CNOT(q0, q1),
        gates.RZ(q1, theta),
        gates.CNOT(q0, q1),
    ]
    return decomp_gate


GATES_TO_DECOMPOSE = {
    "crx": crx_decomp,
    "cry": cry_decomp,
    "crz": crz_decomp,
    "cu1": cu1_decomp,
    "cu3": cu3_decomp,
    "csx": csx_decomp,
    "csxdg": csxdg_decomp,
    "iswap": iswap_decomp,
    "fswap": fswap_decomp,
    "rxx": rxx_decomp,
    "ryy": ryy_decomp,
    "rzz": rzz_decomp,
}


class UnsupportedQiboCircuitError(Exception):
    pass


def decompose_qibo_circuit(qibo_circuit: QiboCircuit) -> QiboCircuit:
    """Returns a QiboCircuit circuit equivalent to the input QiboCircuit
    with all unknown by cirq gates decomposed.

    Args:
        qibo_circuit: QiboCircuit to decompose.

    Returns:
        Decomposed QiboCircuit
    """
    decomp_circuit = QiboCircuit(qibo_circuit.nqubits)
    for gate in qibo_circuit.queue:
        decomposition_func = GATES_TO_DECOMPOSE.get(gate.name, lambda x: x)
        decomp_circuit.add(decomposition_func(gate))

    return decomp_circuit


def from_qibo(qibo_circuit: QiboCircuit) -> Circuit:
    """Returns a Cirq circuit equivalent to the input QiboCircuit.

    Args:
        qibo_circuit: QiboCircuit to convert to a Cirq circuit.

    Returns:
        Cirq circuit representation equivalent to the input QiboCircuit.
    """
    for measurement in qibo_circuit.measurements:
        reg_name = measurement.register_name
        if not reg_name.islower():
            raise UnsupportedQiboCircuitError(
                f"OpenQASM does not support capital letters in "
                f"register names but {reg_name} was used."
            )
    for gate in qibo_circuit.queue:
        if isinstance(gate, gates.M):
            continue
        if gate.is_controlled_by:
            raise UnsupportedQiboCircuitError(
                "OpenQASM does not support multi-controlled gates."
            )
        try:
            gate.qasm_label
        except NotImplementedError:
            raise UnsupportedQiboCircuitError(
                f"{gate.name} is not supported by OpenQASM."
            )

    # Decompose the circuit in known Cirq gates
    qasm = decompose_qibo_circuit(qibo_circuit).to_qasm()

    return cirq_from_qasm(qasm)


def to_qibo(circuit: Circuit) -> QiboCircuit:
    """Returns a QiboCircuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a QiboCircuit.

    Returns:
        QiboCircuit object equivalent to the input Mitiq circuit.
    """
    qasm = cirq_to_qasm(circuit)
    # Remove problematic comment lines in the qasm code
    lines = [
        line.replace(", ", ",")
        for line in qasm.split("\n")
        if not line.strip().startswith("//")
    ]
    # Remove in line problematic comments
    clean_lines = []
    for line in lines:
        clean_line = line.split("//")[0].strip()
        if clean_line:
            clean_lines.append(clean_line)
    qasm = "\n".join(clean_lines)

    return QiboCircuit.from_qasm(qasm)
