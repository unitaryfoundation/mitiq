# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools to determine slack windows in circuits and to insert DDD sequences."""

from collections.abc import Callable
from dataclasses import dataclass

import cirq
import numpy as np
import numpy.typing as npt
from cirq import Circuit, I, LineQubit, synchronize_terminal_measurements

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate
from mitiq.interface.conversions import convert_to_mitiq


@dataclass(frozen=True)
class DDDInfo:
    """Structured information about a DDD insertion pass.

    Attributes:
        num_idle_windows: Number of single-qubit idle windows with slack length
            greater than 1 (candidates for DDD insertion).
        num_sequences_inserted: Number of non-empty DDD sequences that were
            inserted into those windows.
        idle_window_lengths: Length of each candidate idle window, in moments.
    """

    num_idle_windows: int
    num_sequences_inserted: int
    idle_window_lengths: tuple[int, ...]


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
    if not np.issubdtype(mask.dtype, int):
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
    rule: Callable[[int], QPROGRAM],
    *,
    return_info: bool = False,
) -> QPROGRAM | tuple[QPROGRAM, DDDInfo]:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The QPROGRAM circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.
        return_info: If ``False`` (default), return only the circuit with
            DDD sequences added. If ``True``, return a tuple
            ``(circuit_with_ddd, ddd_info)`` where ``ddd_info`` is a
            :class:`~mitiq.ddd.insertion.DDDInfo` describing idle windows
            found and sequences inserted. This is useful for verifying
            that DDD actually modified the circuit (e.g. when there are
            no idle windows).

    Returns:
        The circuit with DDD sequences added, or
        ``(circuit_with_ddd, ddd_info)`` if ``return_info`` is ``True``.
    """
    info_out: list[DDDInfo] = []
    circuit_with_ddd = _insert_ddd_sequences(
        circuit, rule, _info_out=info_out if return_info else None
    )
    if return_info:
        return circuit_with_ddd, info_out[0]
    return circuit_with_ddd


@accept_qprogram_and_validate
def _insert_ddd_sequences(
    circuit: Circuit,
    rule: Callable[[int], QPROGRAM],
    _info_out: list[DDDInfo] | None = None,
) -> Circuit:
    """Returns the circuit with DDD sequences applied according to the input
    rule.

    Args:
        circuit: The Cirq circuit to be modified with DDD sequences.
        rule: The rule determining what DDD sequences should be applied.
            A set of built-in DDD rules can be imported from
            ``mitiq.ddd.rules``.
        _info_out: Optional mutable list used as an out-parameter. When
            provided, a single :class:`DDDInfo` is appended after insertion.
            This keeps the public return type of the decorated function a
            circuit so ``accept_qprogram_and_validate`` conversion is
            unchanged.

    Returns:
        The circuit with DDD sequences added.
    """
    circuit = synchronize_terminal_measurements(circuit)
    if not circuit.are_all_measurements_terminal():
        raise ValueError(
            "This circuit contains midcircuit measurements which "
            "are not currently supported by DDD."
        )

    def cirq_rule(slack_length: int) -> Circuit:
        cirq_circuit, _ = convert_to_mitiq(rule(slack_length))
        qubit_map: dict[cirq.Qid, cirq.Qid] = {
            q: LineQubit(i)
            for i, q in enumerate(sorted(cirq_circuit.all_qubits()))
        }
        return cirq_circuit.transform_qubits(qubit_map)

    slack_matrix = get_slack_matrix_from_circuit_mask(
        _get_circuit_mask(circuit)
    )
    # Copy to avoid mutating the input circuit
    circuit_with_ddd = circuit.copy()
    qubits = sorted(circuit.all_qubits())
    idle_window_lengths: list[int] = []
    num_sequences_inserted = 0
    for moment_idx in range(len(circuit)):
        slack_column = slack_matrix[:, moment_idx]
        for row_index, slack_length in enumerate(slack_column):
            if slack_length > 1:
                idle_window_lengths.append(int(slack_length))
                ddd_sequence = cirq_rule(slack_length).transform_qubits(
                    {LineQubit(0): qubits[row_index]}
                )
                operations = list(ddd_sequence.all_operations())
                if operations:
                    num_sequences_inserted += 1
                for idx, op in enumerate(operations):
                    moment = circuit_with_ddd[moment_idx + idx]
                    op_to_replace = moment.operation_at(*op.qubits)

                    if op_to_replace and op_to_replace.gate == I:
                        moment = moment.without_operations_touching(op.qubits)

                    circuit_with_ddd[moment_idx + idx] = moment.with_operation(
                        op
                    )

    if _info_out is not None:
        _info_out.append(
            DDDInfo(
                num_idle_windows=len(idle_window_lengths),
                num_sequences_inserted=num_sequences_inserted,
                idle_window_lengths=tuple(idle_window_lengths),
            )
        )
    return circuit_with_ddd
