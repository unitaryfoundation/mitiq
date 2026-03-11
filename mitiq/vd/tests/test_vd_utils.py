# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np
import pytest

from mitiq.vd.vd_utils import (
    _apply_cyclic_system_permutation,
    _apply_diagonalizing_gate,
    _apply_symmetric_observable,
    _copy_circuit_parallel,
    _generate_diagonalizing_gate,
)


def test_copy_circuit_parallel_lengths():
    circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)),
        cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(1)),
    )
    for num in range(2, 10):
        new_circuit = _copy_circuit_parallel(circuit, num)
        assert len(new_circuit.all_qubits()) == 2 * num

    circuit = cirq.Circuit(
        cirq.X(cirq.LineQubit(0)),
        cirq.Y(cirq.LineQubit(1)),
        cirq.Z(cirq.LineQubit(2)),
    )
    for num in range(2, 10):
        new_circuit = _copy_circuit_parallel(circuit, num)
        assert len(new_circuit.all_qubits()) == 3 * num


def test_copy_circuit_parallel_full():
    num = 2
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    new_circuit = _copy_circuit_parallel(circuit, num)

    expected_qubits = cirq.LineQubit.range(2 * num)
    expected_circuit = cirq.Circuit(
        cirq.H(expected_qubits[0]),
        cirq.CNOT(expected_qubits[0], expected_qubits[1]),
        cirq.H(expected_qubits[2]),
        cirq.CNOT(expected_qubits[2], expected_qubits[3]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)

    num = 3
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.X(qubits[0]))
    new_circuit = _copy_circuit_parallel(circuit, num)

    expected_qubits = cirq.LineQubit.range(1 * num)
    expected_circuit = cirq.Circuit(
        cirq.X(expected_qubits[0]),
        cirq.X(expected_qubits[1]),
        cirq.X(expected_qubits[2]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)


def test_copy_circuit_parallel_gridqubits():
    num = 3
    qubits = cirq.GridQubit.rect(2, 2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[0], qubits[2]),
    )

    new_circuit = _copy_circuit_parallel(circuit, num)
    new_qubits = new_circuit.all_qubits()

    assert len(new_qubits) == len(circuit.all_qubits()) * num

    expected_qubits = cirq.GridQubit.rect(2 * num, 2)
    expected_circuit = cirq.Circuit(
        cirq.H(expected_qubits[0]),
        cirq.CNOT(expected_qubits[0], expected_qubits[1]),
        cirq.CNOT(expected_qubits[0], expected_qubits[2]),
        cirq.H(expected_qubits[4]),
        cirq.CNOT(expected_qubits[4], expected_qubits[5]),
        cirq.CNOT(expected_qubits[4], expected_qubits[6]),
        cirq.H(expected_qubits[8]),
        cirq.CNOT(expected_qubits[8], expected_qubits[9]),
        cirq.CNOT(expected_qubits[8], expected_qubits[10]),
    )

    new_unitary = cirq.unitary(new_circuit)
    expected_unitary = cirq.unitary(expected_circuit)
    assert np.allclose(new_unitary, expected_unitary)


def test_apply_diagonalizing_gate_keeps_circuit_structure():
    num_copies = 2
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.X(qubits[2]),
    )
    num_qubits = len(circuit.all_qubits())

    copied_circuit = _copy_circuit_parallel(circuit, num_copies)
    new_circuit = _apply_diagonalizing_gate(copied_circuit, num_copies)

    num_ops = len(list(copied_circuit.all_operations()))
    copied_circuit_ops = list(new_circuit.all_operations())

    assert len(copied_circuit_ops) == num_ops + num_qubits
    assert set(new_circuit.all_qubits()) == set(copied_circuit.all_qubits())


@pytest.mark.xfail(reason="VD does not yet support grid qubits")
def test_apply_diagonalizing_gate_fails_on_grid_qubits():
    num_copies = 2
    qubits = cirq.GridQubit.rect(2, 2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.X(qubits[2]),
    )
    num_qubits = len(circuit.all_qubits())

    copied_circuit = _copy_circuit_parallel(circuit, num_copies)
    new_circuit = _apply_diagonalizing_gate(copied_circuit, num_copies)

    num_ops = len(list(copied_circuit.all_operations()))
    copied_circuit_ops = list(new_circuit.all_operations())

    assert len(copied_circuit_ops) == num_ops + num_qubits  # this passes
    assert set(new_circuit.all_qubits()) == set(copied_circuit.all_qubits())


def test_generate_diagonalizing_gate():
    expected_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [0, 0, 0, 1],
        ]
    )

    assert np.allclose(
        _generate_diagonalizing_gate(2)._matrix, expected_matrix
    )


def test_apply_cyclic_system_permutation():
    # Check for ndim=2
    matrix = np.arange(1, 17).reshape(4, 4)
    expected = matrix[[0, 2, 1, 3]]

    result = _apply_cyclic_system_permutation(matrix, N_qubits=1)
    assert np.allclose(result, expected)

    # Check for ndim=3
    tensor = np.arange(1, 65).reshape(4, 4, 4)
    expected = tensor[:, [0, 2, 1, 3], :]

    result = _apply_cyclic_system_permutation(tensor, N_qubits=1)
    assert np.allclose(result, expected)

    # Check for ndim>3
    tensor = np.arange(1, 17).reshape(2, 2, 2, 2)

    with pytest.raises(TypeError, match="matrix should be 2D or 3D ndarray"):
        _apply_cyclic_system_permutation(tensor, N_qubits=1)


def test_edge_case_single_qubit():
    matrix = np.array([[1, 2], [3, 4]])
    # No permutation should happen here
    expected = matrix.copy()
    result = _apply_cyclic_system_permutation(
        matrix, N_qubits=1, num_registers=1
    )
    assert np.allclose(result, expected)


def test_apply_symmetric_observable():
    # Check for ndim=2
    matrix = np.arange(1, 17).reshape(4, 4)
    expected = np.array(
        [
            [
                [1.0, 0.0, 0.0, -4.0],
                [5.0, 0.0, 0.0, -8.0],
                [9.0, 0.0, 0.0, -12.0],
                [13.0, 0.0, 0.0, -16.0],
            ]
        ]
    )

    result = _apply_symmetric_observable(matrix, N_qubits=1)
    assert np.allclose(result, expected)

    # Check for ndim=3
    tensor = np.arange(1, 65).reshape(4, 4, 4)
    expected = np.array(
        [
            [
                [1.0, 0.0, 0.0, -4.0],
                [5.0, 0.0, 0.0, -8.0],
                [9.0, 0.0, 0.0, -12.0],
                [13.0, 0.0, 0.0, -16.0],
            ],
            [
                [17.0, 0.0, 0.0, -20.0],
                [21.0, 0.0, 0.0, -24.0],
                [25.0, 0.0, 0.0, -28.0],
                [29.0, 0.0, 0.0, -32.0],
            ],
            [
                [33.0, 0.0, 0.0, -36.0],
                [37.0, 0.0, 0.0, -40.0],
                [41.0, 0.0, 0.0, -44.0],
                [45.0, 0.0, 0.0, -48.0],
            ],
            [
                [49.0, 0.0, 0.0, -52.0],
                [53.0, 0.0, 0.0, -56.0],
                [57.0, 0.0, 0.0, -60.0],
                [61.0, 0.0, 0.0, -64.0],
            ],
        ]
    )

    result = _apply_symmetric_observable(tensor, N_qubits=1)
    assert np.allclose(result, expected)

    # Check for ndim>3
    tensor = np.arange(1, 17).reshape(2, 2, 2, 2)

    with pytest.raises(ValueError, match="matrix should be 2D or 3D ndarray"):
        _apply_symmetric_observable(tensor, N_qubits=1)


def test_apply_symmetric_observable_X():
    # For N_qubits=1 and M=2, we're dealing with 4x4 matrices
    matrix = np.arange(1, 17).reshape(4, 4)
    observable = np.array([[0, 1], [1, 0]])

    # For N_qubits=1, we expect one output matrix with diagonals scaled
    # by +1 for |0⟩ state and -1 for |1⟩ state
    expected = np.array(
        [
            [
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
            ]
        ]
    )

    result = _apply_symmetric_observable(
        matrix, N_qubits=1, observable=observable
    )
    assert np.allclose(result, expected)
