# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for virtual distillation."""

from unittest.mock import MagicMock

import cirq

from mitiq import MeasurementResult
from mitiq.vd import combine_results, construct_circuits, execute_with_vd


def test_construct_circuits_adds_intended_qubits_and_measurements():
    qubits = cirq.LineQubit.range(6)
    test_circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.Z(qubits[2]),
    )
    vd_circuit = construct_circuits(test_circuit)

    assert vd_circuit.all_qubits() == set(qubits)
    assert len(vd_circuit.all_measurement_key_objs()) == 1


def test_combine_results_gives_correct_number_of_expvals():
    mr = MeasurementResult.from_counts(
        {"0000": 124, "1111": 103, "0001": 89, "1110": 104}
    )
    result = combine_results(mr)

    assert len(result) == 2


def test_execute_with_vd_makes_single_executor_call():
    qubits = cirq.LineQubit.range(6)
    test_circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.Z(qubits[2]),
    )

    mock_executor = MagicMock(return_value=MeasurementResult(["000", "001"]))

    _ = execute_with_vd(test_circuit, mock_executor)

    assert mock_executor.call_count == 1
