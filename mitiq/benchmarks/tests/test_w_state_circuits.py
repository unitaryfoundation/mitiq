# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for W-state benchmarking circuits."""

import cirq
import numpy as np
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.w_state_circuits import generate_w_circuit
from mitiq.utils import _equal


def test_bad_qubit_number():
    for n in (-1, 0):
        with pytest.raises(
            ValueError, match="{} is invalid for the number of qubits. "
        ):
            generate_w_circuit(n)


def test_w4_circuit():
    """Tests for a W-state of 3 qubits."""

    # compare the circuits
    output_circuit = generate_w_circuit(4)
    qubits = cirq.LineQubit.range(4)
    correct_circuit = cirq.Circuit(
        cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 4)))
        .controlled()
        .on(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[0]),
        cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 3)))
        .controlled()
        .on(qubits[1], qubits[2]),
        cirq.CNOT(qubits[2], qubits[1]),
        cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 2)))
        .controlled()
        .on(qubits[2], qubits[3]),
        cirq.CNOT(qubits[3], qubits[2]),
    )
    assert _equal(output_circuit, correct_circuit)

    # compare the state vector
    w4_state_vector = (
        cirq.Simulator()
        .simulate(output_circuit, initial_state=1000)
        .final_state_vector
    )
    correct_final_state_vector = np.array(
        [0, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0]
    )
    assert np.allclose(w4_state_vector, correct_final_state_vector)


def test_w2_circuit():
    """Tests for W-state of 2 qubits."""

    # compare the output circuit with expected circuit
    output_circuit = generate_w_circuit(2)
    qubits = cirq.LineQubit.range(2)
    correct_circuit = cirq.Circuit(
        cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 2)))
        .controlled()
        .on(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[0]),
    )
    assert _equal(output_circuit, correct_circuit)

    # compare the state vector
    w2_state_vector = (
        cirq.Simulator()
        .simulate(output_circuit, initial_state=10)
        .final_state_vector
    )
    correct_final_state_vector = np.array(
        [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]
    )
    assert np.allclose(w2_state_vector, correct_final_state_vector)


def test_w3_circuit():
    """Tests for W-state of 3 qubits."""
    # compare the output circuit with expected circuit
    output_circuit = generate_w_circuit(3)
    qubits = cirq.LineQubit.range(3)
    correct_circuit = cirq.Circuit(
        cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 3)))
        .controlled()
        .on(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[0]),
        cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 2)))
        .controlled()
        .on(qubits[1], qubits[2]),
        cirq.CNOT(qubits[2], qubits[1]),
    )
    assert _equal(output_circuit, correct_circuit)

    # compare the state vector
    w3_state_vector = (
        cirq.Simulator()
        .simulate(output_circuit, initial_state=100)
        .final_state_vector
    )
    correct_final_state_vector = np.array(
        [0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0, 0, 0]
    )
    assert np.allclose(w3_state_vector, correct_final_state_vector)


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_conversion(return_type):
    circuit = generate_w_circuit(3, return_type)
    assert return_type in circuit.__module__
