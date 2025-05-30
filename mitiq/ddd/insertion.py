# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools to determine slack windows in circuits and to insert DDD sequences."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from cirq import Circuit, I, LineQubit, synchronize_terminal_measurements

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate


def _get_circuit_mask(circuit: Circuit) -> npt.NDArray[np.int64]:
    """Given a circuit with n qubits and d moments returns a matrix
    :math:`A` with n rows and d columns. The matrix elements are
    :math:`A_{i,j} = 1` if there is a non-identity gate acting on qubit
    :math:`i` at moment :math:`j`, while :math:`A_{i,j} = 0` otherwise.

    Args:
        circuit: Input circuit to mask with n qubits and d moments

    Returns:
        A mask matrix with n rows and d columns
    """
    qubits = sorted(circuit.all_qubits())
    indexed_qubits = [(i, n) for (i, n) in enumerate(qubits)]
    mask_matrix = np.zeros((len(qubits), len(circuit)), dtype=int)
    for moment_index, moment in enumerate(circuit):
        for op in moment:
            qubit_indices = [
                qubit[0]
                for qubit in indexed_qubits
                if qubit[1] in op.qubits and op.gate != I
            ]
            for qubit_index in qubit_indices:
                mask_matrix[qubit_index, moment_index] = 1
    return mask_matrix


def _validate_integer_matrix(mask: npt.NDArray[np.int64]) -> None:
    """Ensures the input is a NumPy 2d array with integer elements."""
    if not isinstance(mask, np.ndarray):
        raise TypeError("The input matrix must be a numpy.ndarray object.")
    if not np.issubdtype(mask.dtype.type, int):
        raise TypeError("The input matrix must have integer elements.")
    if len(mask.shape) != 2:
        raise ValueError("The input must be a 2-dimensional array.")


def get_slack_matrix_from_circuit_mask(
    mask: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Given a circuit mask matrix :math:`A`, e.g., the output of
    ``_get_circuit_mask()``, returns a slack matrix :math:`B`,
    where :math:`B_{i,j} = t` if the position :math:`A_{i,j}` is the
    initial element of a sequence of :math:`t` zeros (from left to right).

    Args:
        mask: The mask matrix of a quantum circuit.

    Returns:
        The matrix of slack lengths.
    """
    _validate_integer_matrix(mask)
    if not (mask**2 == mask).all():
        raise ValueError("The input matrix elements must be 0 or 1.")

    num_rows, num_cols = mask.shape
    slack_matrix = np.zeros((num_rows, num_cols), dtype=int)
    for r in range(num_rows):
        for c in range(num_cols):
            previous_elem = mask[r, c - 1] if c != 0 else 1
            if previous_elem == 1:
                # Compute slack length
                for elem in mask[r, c::]:
                    if elem == 0:
                        slack_matrix[r, c] += 1
                    else:
                        break

    return slack_matrix


def insert_ddd_sequences(
    circuit: QPROGRAM,
    rule: Callable[[int], Circuit],
) -> QPROGRAM:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The QPROGRAM circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.

    Returns:
        The circuit with DDD sequences added.
    """

    return _insert_ddd_sequences(circuit, rule)


@accept_qprogram_and_validate
def _insert_ddd_sequences(
    circuit: Circuit,
    rule: Callable[[int], Circuit],
) -> Circuit:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The Cirq circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.

    Returns:
        The circuit with DDD sequences added.
    """
    circuit = synchronize_terminal_measurements(circuit)
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            "This circuit contains midcircuit measurements which "
            "are not currently supported by DDD."
        )

    slack_matrix = get_slack_matrix_from_circuit_mask(
        _get_circuit_mask(circuit)
    )
    # Copy to avoid mutating the input circuit
    circuit_with_ddd = circuit.copy()
    qubits = sorted(circuit.all_qubits())
    for moment_idx in range(len(circuit)):
        slack_column = slack_matrix[:, moment_idx]
        for row_index, slack_length in enumerate(slack_column):
            if slack_length > 1:
                ddd_sequence = rule(slack_length).transform_qubits(
                    {LineQubit(0): qubits[row_index]}
                )
                for idx, op in enumerate(ddd_sequence.all_operations()):
                    moment = circuit_with_ddd[moment_idx + idx]
                    op_to_replace = moment.operation_at(*op.qubits)

                    if op_to_replace and op_to_replace.gate == I:
                        moment = moment.without_operations_touching(op.qubits)

                    circuit_with_ddd[moment_idx + idx] = moment.with_operation(
                        op
                    )
    return circuit_with_ddd
