# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for DDD slack windows and DDD insertion tools."""

import cirq
import numpy as np
import pyquil
import pytest
import qiskit

from mitiq.ddd.insertion import (
    DDDInfo,
    _get_circuit_mask,
    get_slack_matrix_from_circuit_mask,
    insert_ddd_sequences,
)
from mitiq.ddd.rules import xx, xyxy

circuit_cirq_one = cirq.Circuit(
    cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
)

qreg_cirq = cirq.GridQubit.rect(10, 1)
circuit_cirq_two = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq),
    cirq.ops.H.on(qreg_cirq[1]),
)

circuit_cirq_three = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq),
    5 * [cirq.ops.H.on(qreg_cirq[1])],
)

circuit_cirq_three_validated = cirq.Circuit(
    cirq.ops.H.on_each(*qreg_cirq),
    5 * [cirq.ops.H.on(qreg_cirq[1])],
    cirq.ops.I.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.X.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.Y.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.X.on_each(qreg_cirq[0], *qreg_cirq[2:]),
    cirq.ops.Y.on_each(qreg_cirq[0], *qreg_cirq[2:]),
)

qreg = qiskit.QuantumRegister(4)
creg = qiskit.ClassicalRegister(4)

# Qiskit test without measurement
circuit_qiskit_one = qiskit.QuantumCircuit(qreg)
for q in qreg:
    circuit_qiskit_one.x(q)
    circuit_qiskit_one.x(3)
circuit_qiskit_one.cx(0, 3)

# Qiskit test with measurement
circuit_qiskit_two = qiskit.QuantumCircuit(qreg, creg)
for q in qreg:
    circuit_qiskit_two.x(q)
    circuit_qiskit_two.x(3)
circuit_qiskit_two.measure(2, 3)
circuit_qiskit_two.cx(0, 3)

circuit_qiskit_validated = qiskit.QuantumCircuit(qreg)
for i in range(4):
    circuit_qiskit_validated.x(i)
    circuit_qiskit_validated.x(3)
    if i != 3 and i != 0:
        circuit_qiskit_validated.id(i)
        circuit_qiskit_validated.x(i)
        circuit_qiskit_validated.id(i)
        circuit_qiskit_validated.x(i)
        circuit_qiskit_validated.id(i)
    elif i == 0:
        circuit_qiskit_validated.id(i)
        circuit_qiskit_validated.x(i)
        circuit_qiskit_validated.x(i)
        circuit_qiskit_validated.id(i)
circuit_qiskit_validated.cx(0, 3)

# Qiskit validate with measurement
circuit_qiskit_two_validated = qiskit.QuantumCircuit(qreg, creg)

for i in range(4):
    circuit_qiskit_two_validated.x(i)
    circuit_qiskit_two_validated.x(3)
    if i != 3 and i != 0:
        if i == 1:
            circuit_qiskit_two_validated.id(i)
        circuit_qiskit_two_validated.id(i)
        circuit_qiskit_two_validated.x(i)
        circuit_qiskit_two_validated.id(i)
        circuit_qiskit_two_validated.x(i)
        circuit_qiskit_two_validated.id(i)
    elif i == 0:
        circuit_qiskit_two_validated.id(i)
        circuit_qiskit_two_validated.x(i)
        circuit_qiskit_two_validated.x(i)
        circuit_qiskit_two_validated.id(i)
circuit_qiskit_two_validated.cx(0, 3)
circuit_qiskit_two_validated.measure(2, 3)

# Define test mask matrices
test_mask_one = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)

test_mask_two = np.array(
    [
        [1, 0],
        [1, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
    ]
)

one_mask = np.array(
    [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ]
)
two_mask = np.array(
    [
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)
mixed_mask = np.array(
    [
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]
)
# Define test slack matrices
one_slack = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ]
)
two_slack = np.array(
    [
        [2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [2, 0, 0, 2, 0],
    ]
)
mixed_slack = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 4, 0, 0, 0],
        [5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
masks = [one_mask, two_mask, mixed_mask]
slack_matrices = [one_slack, two_slack, mixed_slack]


@pytest.mark.parametrize(
    ("circuit", "test_mask"),
    [(circuit_cirq_one, test_mask_one), (circuit_cirq_two, test_mask_two)],
)
def test_get_circuit_mask(circuit, test_mask):
    circuit_mask = _get_circuit_mask(circuit)
    assert np.allclose(circuit_mask, test_mask)


def test_get_slack_matrix_from_circuit_mask():
    for mask, expected in zip(masks, slack_matrices):
        slack_matrix = get_slack_matrix_from_circuit_mask(mask)
        assert np.allclose(slack_matrix, expected)


def test_get_slack_matrix_from_circuit_mask_extreme_cases():
    assert np.allclose(
        get_slack_matrix_from_circuit_mask(np.array([[0]])), np.array([[1]])
    )
    assert np.allclose(
        get_slack_matrix_from_circuit_mask(np.array([[1]])), np.array([[0]])
    )


def test_get_slack_matrix_from_circuit__bad_input_errors():
    with pytest.raises(TypeError, match="must be a numpy"):
        get_slack_matrix_from_circuit_mask([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="must be a 2-dimensional"):
        get_slack_matrix_from_circuit_mask(np.array([1, 2]))
    with pytest.raises(TypeError, match="must have integer elements"):
        get_slack_matrix_from_circuit_mask(np.array([[1, 0], [1, 1.7]]))
    with pytest.raises(ValueError, match="elements must be 0 or 1"):
        get_slack_matrix_from_circuit_mask(np.array([[2, 0], [0, 0]]))


@pytest.mark.parametrize(
    ("circuit", "result", "rule"),
    [
        (circuit_cirq_two, circuit_cirq_two, xx),
        (circuit_cirq_three, circuit_cirq_three_validated, xyxy),
        (circuit_qiskit_one, circuit_qiskit_validated, xx),
        (circuit_qiskit_two, circuit_qiskit_two_validated, xx),
    ],
)
def test_insert_sequences(circuit, result, rule):
    circuit_with_sequences = insert_ddd_sequences(circuit, rule)
    assert circuit_with_sequences == result


def test_midcircuit_measurement_raises_error():
    qreg = qiskit.QuantumRegister(2)
    creg = qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    for q in qreg:
        circuit.x(q)

    circuit.measure(0, 0)
    circuit.cx(0, 1)

    with pytest.raises(ValueError, match="midcircuit measurements"):
        insert_ddd_sequences(circuit, xx)


def test_pyquil_midcircuit_measurement_raises_error():
    p = pyquil.Program()
    cbit = p.declare("cbit")
    p += pyquil.gates.X(0)
    p += pyquil.gates.X(1)
    p += pyquil.gates.MEASURE(0, cbit[0])
    p += pyquil.gates.X(0)

    with pytest.raises(ValueError, match="midcircuit measurements"):
        insert_ddd_sequences(p, xx)


def test_insert_sequence_over_identity_gates():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )
    circuit_expected = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.X.on_each(*qubits),
        cirq.ops.X.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )

    ddd_circuit = insert_ddd_sequences(circuit, rule=xx)

    assert ddd_circuit == circuit_expected


def test_insert_sequences_with_qiskit_rule():
    """DDD rules that return non-Cirq circuits (e.g. Qiskit) should work."""
    from mitiq.interface.mitiq_qiskit.conversions import to_qiskit

    def qiskit_xx(slack_length: int) -> qiskit.QuantumCircuit:
        cirq_seq = xx(slack_length)
        return to_qiskit(cirq_seq)

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )
    expected = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.X.on_each(*qubits),
        cirq.ops.X.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )

    result = insert_ddd_sequences(circuit, rule=qiskit_xx)
    assert result == expected


def test_insert_ddd_sequences_return_info_default_is_circuit_only():
    """Default API remains circuit-only (no tuple)."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )
    result = insert_ddd_sequences(circuit, rule=xx)
    assert isinstance(result, cirq.Circuit)
    assert not isinstance(result, tuple)


def test_insert_ddd_sequences_return_info_true():
    """Optional return_info reports idle windows and insertions."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )
    circuit_with_ddd, info = insert_ddd_sequences(
        circuit, rule=xx, return_info=True
    )

    assert isinstance(circuit_with_ddd, cirq.Circuit)
    assert isinstance(info, DDDInfo)
    assert info.num_idle_windows == 2
    assert info.num_sequences_inserted == 2
    assert info.idle_window_lengths == (2, 2)
    assert circuit_with_ddd == insert_ddd_sequences(circuit, rule=xx)


def test_insert_ddd_sequences_return_info_no_idle_windows():
    """When there are no insertable idle windows, info reports zeros."""
    q = cirq.LineQubit(0)
    # No multi-moment idle windows: consecutive non-identity ops only.
    circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.H(q))
    circuit_with_ddd, info = insert_ddd_sequences(
        circuit, rule=xx, return_info=True
    )

    assert circuit_with_ddd == circuit
    assert info.num_idle_windows == 0
    assert info.num_sequences_inserted == 0
    assert info.idle_window_lengths == ()


def test_insert_ddd_sequences_return_info_empty_rule():
    """Idle windows counted even if the rule inserts nothing."""

    def empty_rule(slack_length: int) -> cirq.Circuit:
        return cirq.Circuit()

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.ops.H.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.I.on_each(*qubits),
        cirq.ops.H.on_each(*qubits),
    )
    circuit_with_ddd, info = insert_ddd_sequences(
        circuit, rule=empty_rule, return_info=True
    )

    assert circuit_with_ddd == circuit
    assert info.num_idle_windows == 2
    assert info.num_sequences_inserted == 0
    assert info.idle_window_lengths == (2, 2)


def test_insert_ddd_sequences_return_info_qiskit():
    """return_info works through frontend conversion (Qiskit)."""
    qreg = qiskit.QuantumRegister(2)
    circuit = qiskit.QuantumCircuit(qreg)
    circuit.h(0)
    circuit.h(1)
    circuit.id(0)
    circuit.id(1)
    circuit.id(0)
    circuit.id(1)
    circuit.h(0)
    circuit.h(1)

    result, info = insert_ddd_sequences(circuit, rule=xx, return_info=True)
    assert isinstance(result, qiskit.QuantumCircuit)
    assert info.num_idle_windows >= 1
    assert info.num_sequences_inserted == info.num_idle_windows
