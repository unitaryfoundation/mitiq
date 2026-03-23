# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for TREX utility functions."""

import cirq
import numpy as np
import pytest

from mitiq import MeasurementResult
from mitiq.experimental.trex.trex_utils import (
    create_calibration_circuit,
    insert_x_before_first_measurement,
    xor_bitstrings,
)

# --- Tests for insert_x_before_first_measurement ---


def test_insert_x_no_flips():
    """With all-zero bitstring, circuit is unchanged (copy returned)."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits),
    )
    bitstring = np.array([0, 0], dtype=np.int64)
    result = insert_x_before_first_measurement(circuit, bitstring, qubits)

    assert result == circuit
    assert result is not circuit  # Should be a copy


def test_insert_x_single_qubit():
    """X gate inserted before measurement on specified qubit."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.measure(*qubits),
    )
    bitstring = np.array([1, 0], dtype=np.int64)
    result = insert_x_before_first_measurement(circuit, bitstring, qubits)

    # Should have one more moment (the X gate moment before measurement)
    assert len(result) == len(circuit) + 1

    # Verify X gate is present before measurement
    ops = list(result.all_operations())
    x_ops = [op for op in ops if op.gate == cirq.X]
    assert len(x_ops) == 1
    assert x_ops[0].qubits == (qubits[0],)


def test_insert_x_all_qubits():
    """X gates inserted on all qubits."""
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(*qubits))
    bitstring = np.array([1, 1, 1], dtype=np.int64)
    result = insert_x_before_first_measurement(circuit, bitstring, qubits)

    # Should have original moment + X moment
    assert len(result) == 2

    x_ops = [op for op in result.all_operations() if op.gate == cirq.X]
    assert len(x_ops) == 3


def test_insert_x_length_mismatch():
    """Raises ValueError if bitstring and qubits have different lengths."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(*qubits))
    bitstring = np.array([1, 0, 1], dtype=np.int64)  # Length 3, qubits is 2

    with pytest.raises(ValueError, match="Length of bitstring"):
        insert_x_before_first_measurement(circuit, bitstring, qubits)


def test_insert_x_preserves_unitary_with_flip():
    """Inserting X + measurement then XOR should recover original outcome
    in noiseless case."""
    qubits = cirq.LineQubit.range(2)
    # Circuit that prepares |11>
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.X(qubits[1]),
        cirq.measure(*qubits),
    )
    bitstring = np.array([1, 1], dtype=np.int64)
    twirled = insert_x_before_first_measurement(circuit, bitstring, qubits)

    # Simulate the twirled circuit
    sim = cirq.DensityMatrixSimulator()
    result = sim.run(twirled, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))

    # All results should be [0, 0] because X flips |1> to |0>
    assert np.all(bitstrings == 0)

    # After XOR with bitstring, should recover [1, 1]
    mr = MeasurementResult(bitstrings.tolist(), qubit_indices=(0, 1))
    flipped = xor_bitstrings(mr, bitstring)
    assert np.all(flipped._bitstrings == 1)


# --- Tests for create_calibration_circuit ---


def test_calibration_circuit_no_flips():
    """Calibration with no X gates: just measure |0...0>."""
    qubits = cirq.LineQubit.range(2)
    bitstring = np.array([0, 0], dtype=np.int64)
    circuit = create_calibration_circuit(qubits, bitstring)

    # Should only have a measurement gate
    ops = list(circuit.all_operations())
    assert len(ops) == 1
    assert cirq.is_measurement(ops[0])


def test_calibration_circuit_with_flips():
    """Calibration circuit has X gates on specified qubits."""
    qubits = cirq.LineQubit.range(3)
    bitstring = np.array([1, 0, 1], dtype=np.int64)
    circuit = create_calibration_circuit(qubits, bitstring)

    x_ops = [op for op in circuit.all_operations() if op.gate == cirq.X]
    assert len(x_ops) == 2
    assert qubits[0] in x_ops[0].qubits or qubits[0] in x_ops[1].qubits
    assert qubits[2] in x_ops[0].qubits or qubits[2] in x_ops[1].qubits


def test_calibration_circuit_simulation():
    """Simulate calibration circuit and verify XOR recovers zeros."""
    qubits = cirq.LineQubit.range(2)
    bitstring = np.array([1, 1], dtype=np.int64)
    circuit = create_calibration_circuit(qubits, bitstring)

    sim = cirq.DensityMatrixSimulator()
    result = sim.run(circuit, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))

    # With X on both, measurement should give [1, 1]
    assert np.all(bitstrings == 1)

    # After XOR, should give [0, 0]
    mr = MeasurementResult(bitstrings.tolist(), qubit_indices=(0, 1))
    flipped = xor_bitstrings(mr, bitstring)
    assert np.all(flipped._bitstrings == 0)


def test_calibration_circuit_length_mismatch():
    """Raises ValueError if bitstring and qubits have different lengths."""
    qubits = cirq.LineQubit.range(2)
    bitstring = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="Length of bitstring"):
        create_calibration_circuit(qubits, bitstring)


# --- Tests for xor_bitstrings ---


def test_xor_all_zeros():
    """XOR with all zeros leaves bitstrings unchanged."""
    mr = MeasurementResult(["01", "10", "11"], qubit_indices=(0, 1))
    bitstring = np.array([0, 0], dtype=np.int64)
    result = xor_bitstrings(mr, bitstring)

    assert np.array_equal(result.asarray, mr.asarray)


def test_xor_all_ones():
    """XOR with all ones flips all bits."""
    mr = MeasurementResult(["00", "01", "10", "11"], qubit_indices=(0, 1))
    bitstring = np.array([1, 1], dtype=np.int64)
    result = xor_bitstrings(mr, bitstring)

    expected = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    assert np.array_equal(result.asarray, expected)


def test_xor_partial():
    """XOR flips only specified bits."""
    mr = MeasurementResult(["000", "111"], qubit_indices=(0, 1, 2))
    bitstring = np.array([1, 0, 1], dtype=np.int64)
    result = xor_bitstrings(mr, bitstring)

    expected = np.array([[1, 0, 1], [0, 1, 0]])
    assert np.array_equal(result.asarray, expected)


def test_xor_preserves_qubit_indices():
    """XOR preserves qubit_indices from the original result."""
    mr = MeasurementResult(["01", "10"], qubit_indices=(2, 5))
    bitstring = np.array([1, 0], dtype=np.int64)
    result = xor_bitstrings(mr, bitstring)

    assert result.qubit_indices == (2, 5)


def test_xor_length_mismatch():
    """Raises ValueError if bitstring length doesn't match nqubits."""
    mr = MeasurementResult(["01", "10"], qubit_indices=(0, 1))
    bitstring = np.array([1, 0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="Length of bitstring"):
        xor_bitstrings(mr, bitstring)


def test_insert_x_no_measurement():
    """X gates appended at end when circuit has no measurement."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]))
    bitstring = np.array([1, 0], dtype=np.int64)
    result = insert_x_before_first_measurement(circuit, bitstring, qubits)

    # X gate should be appended at the end
    x_ops = [op for op in result.all_operations() if op.gate == cirq.X]
    assert len(x_ops) == 1
    assert x_ops[0].qubits == (qubits[0],)


def test_xor_double_application_is_identity():
    """Applying XOR twice with the same bitstring is identity."""
    mr = MeasurementResult(["01", "10", "11"], qubit_indices=(0, 1))
    bitstring = np.array([1, 1], dtype=np.int64)

    result = xor_bitstrings(xor_bitstrings(mr, bitstring), bitstring)
    assert np.array_equal(result.asarray, mr.asarray)
