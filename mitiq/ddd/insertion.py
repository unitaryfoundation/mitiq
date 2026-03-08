# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools to determine slack windows in circuits and to insert DDD sequences."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from cirq import Circuit, I, LineQubit, synchronize_terminal_measurements

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate
from mitiq.interface.conversions import convert_to_mitiq, convert_from_mitiq


@dataclass
class DDDInfo:
    """Information about DDD sequence insertion.
    
    Attributes:
        num_idle_windows: Number of idle windows found in the circuit.
        num_sequences_inserted: Number of DDD sequences actually inserted.
        idle_window_lengths: List of lengths of each idle window.
    """
    num_idle_windows: int
    num_sequences_inserted: int
    idle_window_lengths: list[int]


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


def _insert_ddd_sequences_with_info(
    circuit: Circuit,
    rule: Callable[[int], Circuit],
) -> tuple[Circuit, DDDInfo]:
    """Internal function that returns circuit with DDD sequences and info.
    
    Args:
        circuit: The Cirq circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.

    Returns:
        A tuple (circuit_with_ddd, ddd_info).
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
    
    # Track insertion statistics
    num_idle_windows = 0
    num_sequences_inserted = 0
    idle_window_lengths: list[int] = []
    
    for moment_idx in range(len(circuit)):
        slack_column = slack_matrix[:, moment_idx]
        for row_index, slack_length in enumerate(slack_column):
            if slack_length > 1:
                num_idle_windows += 1
                idle_window_lengths.append(slack_length)
                ddd_sequence = rule(slack_length).transform_qubits(
                    {LineQubit(0): qubits[row_index]}
                )
                sequence_inserted = False
                for idx, op in enumerate(ddd_sequence.all_operations()):
                    moment = circuit_with_ddd[moment_idx + idx]
                    op_to_replace = moment.operation_at(*op.qubits)

                    if op_to_replace and op_to_replace.gate == I:
                        moment = moment.without_operations_touching(op.qubits)

                    circuit_with_ddd[moment_idx + idx] = moment.with_operation(
                        op
                    )
                    sequence_inserted = True
                if sequence_inserted:
                    num_sequences_inserted += 1
    
    ddd_info = DDDInfo(
        num_idle_windows=num_idle_windows,
        num_sequences_inserted=num_sequences_inserted,
        idle_window_lengths=idle_window_lengths,
    )
    
    return circuit_with_ddd, ddd_info


def insert_ddd_sequences(
    circuit: QPROGRAM,
    rule: Callable[[int], Circuit],
    *,
    return_info: bool = False,
    verbose: bool = False,
) -> QPROGRAM | tuple[QPROGRAM, DDDInfo]:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The QPROGRAM circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.
        return_info: If True, return a tuple ``(circuit, ddd_info)`` where
            ``ddd_info`` is a ``DDDInfo`` object containing details about
            the insertion (number of idle windows, sequences inserted, etc.).
        verbose: If True, print information about the DDD insertion.

    Returns:
        The circuit with DDD sequences added, or a tuple ``(circuit, ddd_info)``
        if ``return_info=True``.
    """
    # Convert to Mitiq internal representation
    mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)
    
    # Insert DDD sequences
    circuit_with_ddd, ddd_info = _insert_ddd_sequences_with_info(
        mitiq_circuit, rule
    )
    
    if verbose:
        print(f"DDD: found {ddd_info.num_idle_windows} idle windows; inserted {ddd_info.num_sequences_inserted} sequences")
    
    # Convert back to original type
    result_circuit = convert_from_mitiq(circuit_with_ddd, input_circuit_type)
    
    if return_info:
        return result_circuit, ddd_info
    return result_circuit


# Keep the old decorated version for backward compatibility
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
    circuit_with_ddd, _ = _insert_ddd_sequences_with_info(circuit, rule)
    return circuit_with_ddd
