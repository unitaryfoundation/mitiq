# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import cirq
import numpy as np
from numpy.typing import NDArray

from mitiq.observable import Observable


def _copy_circuit_parallel(
    circuit: cirq.Circuit, num_copies: int = 2
) -> cirq.Circuit:
    """Given a circuit that acts on N qubits, this function returns a circuit
    that copies the circuit num_copies times in parallel. This means the
    resulting circuit has N * num_copies qubits.

    Args:
        circuit: The circuit to be copied.
        num_copies: The number of copies of circuit to be made.

    Returns:
        A cirq circuit that is the parallel composition of
          num_copies copies of circuit.
    """

    new_circuit = cirq.Circuit()
    N = len(circuit.all_qubits())
    qubits = list(circuit.all_qubits())

    # LineQubits
    if isinstance(qubits[0], cirq.LineQubit):

        def map_for_line_qubits(q: cirq.Qid) -> cirq.Qid:
            assert isinstance(q, cirq.LineQubit)
            return cirq.LineQubit(q.x + N * i)

        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(map_for_line_qubits)

    # GridQubits
    elif isinstance(qubits[0], cirq.GridQubit):
        qubits_cast_grid = cast(List[cirq.GridQubit], qubits)
        grid_rows = max([qu.row + 1 for qu in qubits_cast_grid])

        def map_for_grid_qubits(qu: cirq.Qid) -> cirq.Qid:
            assert isinstance(qu, cirq.GridQubit)
            return cirq.GridQubit(qu.row + grid_rows * i, qu.col)

        for i in range(num_copies):
            new_circuit += circuit.transform_qubits(map_for_grid_qubits)

    return new_circuit


def _apply_diagonalizing_gate(
    circuit: cirq.Circuit, num_copies: int
) -> cirq.Circuit:
    """Apply the diagonalizing gate to a circuit, assuming the circuit has been
    copied in parallel ``num_copies`` times.

    The diagonalizing gate is applied to the first qubit of all copies, to the
    second qubit of all copies, etc.

    Args:
        circuit: The circuit to apply the diagonalizing gate to.
        num_copies: The number of copies of the original circuit that the input
            circuit consists of.

    Returns:
        The circuit with the diagonalizing gate applied.
    """

    new_circuit = circuit.copy()

    # this is the number of qubits in the original circuit
    N = len(circuit.all_qubits()) // num_copies

    diag_gate = _generate_diagonalizing_gate(num_copies)

    for i in range(N):
        qubits = [
            cirq.LineQubit(i + N * j) for j in range(num_copies)
        ]  # select qubit i of each copy

        new_circuit.append(diag_gate(*qubits))

    return new_circuit


def _generate_diagonalizing_gate(num_copies: int = 2) -> cirq.Gate:
    """Generate the diagonalizing gate for the VD algorithm. Currently only
    ``num_copies=2`` is supported.

    Args:
        num_qubits: The number of qubits that the gate acts on.

    Returns: The diagonalizing gate.
    """
    if num_copies == 2:
        diagonalizing_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise NotImplementedError(
            "Only num_copies = 2 is currently supported."
        )

    return cirq.MatrixGate(diagonalizing_matrix)


def _apply_cyclic_system_permutation(
    matrix: NDArray[np.complex64], N_qubits: int, num_registers: int = 2
) -> NDArray[np.complex64]:
    """
    Function that shifts the rows of a matrix or vector in such a way,
    that each of the num_registers registers of N_qubit qubits are shifted
    cyclically.The implementation is identical to left multiplication
    with repeated swap gates, however this optimisation in considerably
    faster.

    Args:
        matrix: The matrix or vector that should be shifted.
        N_qubits: The number of qubits in each register.
        num_registers: The number of registers.

    Returns:
        The matrix or vector with the rows shifted cyclically.
    """
    matrix = np.array(matrix)

    # determine the row permutation for the cyclic shift operation
    permutation = [
        j + i
        for j in range(2**N_qubits)
        for i in range(0, 2 ** (num_registers * N_qubits), 2**N_qubits)
    ]

    # Some fancy index magic to permute the rows in O(n) time
    # and space (n=2**(num_registers*N_qubits))
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))

    if matrix.ndim == 2:
        matrix = matrix[idx]
    elif matrix.ndim == 3:
        matrix[:] = matrix[:, idx]
    else:
        raise TypeError("matrix should be 2D or 3D ndarray")
    return matrix


def _apply_symmetric_observable(
    matrix: NDArray[np.complex64],
    N_qubits: int,
    observable: Optional[Union[Observable, NDArray[np.complex64]]] = None,
) -> NDArray[np.complex64]:
    """
    Function that applies a symmetric observable to a matrix or vector.

    Args:
        matrix: The matrix or vector that should be shifted.
        N_qubits: The number of qubits in each register.
        observable: The observable that should be applied.
            If None, the Z observable is used.
        num_registers: The number of registers.

    Returns:
        The matrix or vector with the observable applied.
    """
    z_matrix = np.array([[1.0, 0.0], [0.0, -1.0]])

    if observable is None or (
        isinstance(observable, np.ndarray)
        and np.allclose(observable, z_matrix)
    ):
        # use the default Z observable
        sym_observable_diagonals: List[NDArray[np.complex64]] = []
        for i in range(N_qubits):
            observable_i_diagonal = np.array(
                [
                    j
                    for _ in range(2 ** (i))
                    for j in [1.0, -1.0]
                    for _ in range(2 ** (N_qubits - i - 1))
                ]
            )

            # turn [a, b, c] into [a,a,a,b,b,b,c,c,c]. This is the same as
            # tensoring the N_qubit identity on the right
            observable_i_diagonal_system1 = np.array(
                [observable_i_diagonal for _ in range(2**N_qubits)]
            ).flatten("F")
            # turn [a,b,c] into [a,b,c,a,b,c,a,b,c]. This is the same as
            # tensoring the N_qubit identity on the left
            observable_i_diagonal_system2 = np.array(
                [observable_i_diagonal for _ in range(2**N_qubits)]
            ).flatten("C")
            # add the symmetric observable
            # Create the combined diagonal and explicitly convert to complex64
            combined_diagonal = (
                observable_i_diagonal_system1 + observable_i_diagonal_system2
            ) / 2
            sym_observable_diagonals.append(
                combined_diagonal.astype(np.complex64)
            )

        if matrix.ndim == 2:
            return np.array([sod * matrix for sod in sym_observable_diagonals])
        elif matrix.ndim == 3:
            return np.array(
                [
                    sod * mat
                    for sod in sym_observable_diagonals
                    for mat in matrix
                ]
            )
        else:
            raise ValueError("matrix should be 2D or 3D ndarray")

    else:
        obs_array = (
            observable
            if isinstance(observable, np.ndarray)
            else observable.matrix()
        )
        sym_observable_matrices: List[NDArray[np.complex64]] = []
        for i in range(N_qubits):
            observable_i_matrix = np.kron(
                np.kron(np.eye(2**i), obs_array),
                np.eye(2 ** (N_qubits - i - 1)),
            )
            sym_observable_matrix = (
                np.kron(observable_i_matrix, np.eye(2**N_qubits))
                + np.kron(np.eye(2**N_qubits), observable_i_matrix)
            ) / 2
            sym_observable_matrices.append(
                sym_observable_matrix.astype(np.complex64)
            )

        return np.array(sym_observable_matrices) @ matrix
