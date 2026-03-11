# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qibo <-> Cirq conversions."""

import cirq
import numpy as np
import pytest
import qibo
from qibo.models.circuit import Circuit as QiboCircuit

from mitiq.interface.mitiq_qibo import (
    UnsupportedQiboCircuitError,
    from_qibo,
    to_qibo,
)
from mitiq.utils import _equal


def test_from_qibo():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.CNOT(0, 1))
    qibo_circuit.add(qibo.gates.M(0))

    circuit = from_qibo(qibo_circuit)

    correct = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    correct.append(cirq.measure(cirq.LineQubit(0)))

    assert _equal(circuit, correct, require_qubit_equality=False)


def test_from_qibo_register_name_error():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.CNOT(0, 1))
    qibo_circuit.add(qibo.gates.M(0, register_name="K"))

    with pytest.raises(
        UnsupportedQiboCircuitError,
        match=(
            "OpenQASM does not support capital letters in "
            "register names but K was used."
        ),
    ):
        from_qibo(qibo_circuit)


def test_from_qibo_unsupported_multi_controlled_gate():
    qibo_circuit = QiboCircuit(4)
    qibo_circuit.add(qibo.gates.X(0).controlled_by(1, 2, 3))
    with pytest.raises(
        UnsupportedQiboCircuitError,
        match="OpenQASM does not support multi-controlled gates.",
    ):
        from_qibo(qibo_circuit)


def test_from_qibo_unsupported_gate():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.fSim(0, 1, 0.4, 0.4))
    with pytest.raises(
        UnsupportedQiboCircuitError,
        match="fsim is not supported by OpenQASM.",
    ):
        from_qibo(qibo_circuit)


def test_from_qibo_unknown_cirq_gate():
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibo.gates.CRY(0, 1, 0.4))
    qibo_circuit.add(qibo.gates.M(1))
    circuit = from_qibo(qibo_circuit)

    q0, q1 = cirq.LineQubit.range(2)
    correct = cirq.Circuit()
    correct.append(cirq.Ry(rads=0.2).on(q1))
    correct.append(cirq.CNOT(q0, q1))
    correct.append(cirq.Ry(rads=-0.2).on(q1))
    correct.append(cirq.CNOT(q0, q1))
    correct.append(cirq.measure(q1))

    assert _equal(circuit, correct, require_qubit_equality=False)


@pytest.mark.parametrize("random_state", range(10))
def test_to_from_qibo(random_state):
    circuit = cirq.testing.random_circuit(
        qubits=4, n_moments=2, op_density=1, random_state=random_state
    )

    converted = from_qibo(to_qibo(circuit))
    # Gates (e.g. iSWAP) aren't guaranteed to be preserved. Check unitary
    # instead of circuit equality.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(converted), cirq.unitary(circuit), atol=1e-7
    )


def test_to_from_qibo_cnot_same_gates():
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(*qreg))
    converted = from_qibo(to_qibo(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)


def test_to_from_qibo_identity():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q))
    # Identity gate
    converted = from_qibo(to_qibo(circuit))
    assert _equal(circuit, converted, require_qubit_equality=False)


@pytest.mark.parametrize("i", range(10))
def test_qibo_integration(i):
    gates = [
        qibo.gates.X(0),
        qibo.gates.Y(0),
        qibo.gates.Z(0),
        qibo.gates.S(0),
        qibo.gates.SDG(0),
        qibo.gates.T(0),
        qibo.gates.I(0),
        qibo.gates.TDG(0),
        qibo.gates.RX(0, 0.4),
        qibo.gates.RY(0, 0.4),
        qibo.gates.RZ(0, 0.4),
        qibo.gates.H(0),
        qibo.gates.SX(0),
        qibo.gates.SXDG(0),
        qibo.gates.CSX(0, 1),
        qibo.gates.CSXDG(0, 1),
        qibo.gates.TOFFOLI(0, 1, 2),
        qibo.gates.SWAP(0, 1),
        qibo.gates.iSWAP(0, 1),
        qibo.gates.FSWAP(0, 1),
        qibo.gates.CNOT(0, 1),
        qibo.gates.CZ(0, 1),
        qibo.gates.U1(0, 0.4),
        qibo.gates.U2(0, 0.4, 0.5),
        qibo.gates.U3(0, 0.4, 0.5, 0.6),
        qibo.gates.CU1(0, 1, 0.4),
        qibo.gates.CU3(0, 1, 0.4, 0.5, 0.6),
        qibo.gates.CRX(0, 1, 0.4),
        qibo.gates.CRY(0, 1, 0.4),
        qibo.gates.CRZ(0, 1, 0.4),
        qibo.gates.RXX(0, 1, 0.4),
        qibo.gates.RYY(0, 1, 0.4),
        qibo.gates.RZZ(0, 1, 0.4),
    ]

    layers = 3
    np.random.seed(13 * i)
    gates_per_layers = [np.random.permutation(gates) for _ in range(layers)]

    qibo_circuit = QiboCircuit(3)
    for gates in gates_per_layers:
        for gate in gates:
            qibo_circuit.add(gate)

    base_circ = from_qibo(qibo_circuit)
    qibo_recovered = to_qibo(base_circ)
    circ_recovered = from_qibo(qibo_recovered)
    u_1 = cirq.unitary(base_circ)
    u_2 = cirq.unitary(circ_recovered)
    cirq.testing.assert_allclose_up_to_global_phase(u_1, u_2, atol=0)
