# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qrisp <-> Cirq conversions."""

import cirq
from qrisp import QuantumCircuit as QrispCircuit

from mitiq.interface.mitiq_qrisp import from_qrisp, to_qrisp
from mitiq.utils import _equal


def test_from_qrisp():
    qrisp_circuit = QrispCircuit(2)
    qrisp_circuit.cx(0, 1)
    qrisp_circuit.measure(0)

    circuit = from_qrisp(qrisp_circuit)

    correct = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    correct.append(cirq.measure(cirq.LineQubit(0)))

    assert _equal(circuit, correct, require_qubit_equality=False)


def test_to_qrisp():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.measure(q0, q1))

    qrisp_circuit = to_qrisp(circuit)

    assert isinstance(qrisp_circuit, QrispCircuit)
    assert qrisp_circuit.num_qubits() == 2


def test_to_from_qrisp():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.Y(q2))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.CZ(q1, q2))
    circuit.append(cirq.measure(q0, q1, q2))

    converted = from_qrisp(to_qrisp(circuit))

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(converted), cirq.unitary(circuit), atol=1e-7
    )


def test_to_from_qrisp_cnot():
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(*qreg))
    circuit.append(cirq.measure(*qreg))
    converted = from_qrisp(to_qrisp(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)


def test_from_qrisp_to_qrisp():
    qrisp_circuit = QrispCircuit(2)
    qrisp_circuit.h(0)
    qrisp_circuit.cx(0, 1)
    qrisp_circuit.measure(0)
    qrisp_circuit.measure(1)

    circuit = from_qrisp(qrisp_circuit)
    qrisp_recovered = to_qrisp(circuit)
    circuit_recovered = from_qrisp(qrisp_recovered)

    u_1 = cirq.unitary(circuit)
    u_2 = cirq.unitary(circuit_recovered)
    cirq.testing.assert_allclose_up_to_global_phase(u_1, u_2, atol=1e-7)


def test_qrisp_integration():
    qubits = 3
    qrisp_circuit = QrispCircuit(qubits)
    qrisp_circuit.x(0)
    qrisp_circuit.y(1)
    qrisp_circuit.z(2)
    qrisp_circuit.h(0)
    qrisp_circuit.s(1)
    qrisp_circuit.t(2)
    qrisp_circuit.rx(0.4, 0)
    qrisp_circuit.ry(0.4, 1)
    qrisp_circuit.rz(0.4, 2)
    qrisp_circuit.cx(0, 1)
    qrisp_circuit.cz(0, 2)
    qrisp_circuit.swap(0, 1)
    qrisp_circuit.ccx(0, 1, 2)
    qrisp_circuit.measure(0)
    qrisp_circuit.measure(1)
    qrisp_circuit.measure(2)

    base_circ = from_qrisp(qrisp_circuit)
    qrisp_recovered = to_qrisp(base_circ)
    circ_recovered = from_qrisp(qrisp_recovered)
    u_1 = cirq.unitary(base_circ)
    u_2 = cirq.unitary(circ_recovered)
    cirq.testing.assert_allclose_up_to_global_phase(u_1, u_2, atol=0)
